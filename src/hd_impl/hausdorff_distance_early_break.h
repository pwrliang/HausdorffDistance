#ifndef HAUSDORFF_DISTANCE_EARLY_BREAK_H
#define HAUSDORFF_DISTANCE_EARLY_BREAK_H
#include <glog/logging.h>
#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <thrust/shuffle.h>

#include <atomic>
#include <thread>
#include <vector>

#include "geoms/distance.h"
#include "hausdorff_distance.h"
#include "utils/array_view.h"
#include "utils/derived_atomic_functions.h"
#include "utils/shared_value.h"
#include "utils/stopwatch.h"
#include "utils/stream.h"
#include "utils/type_traits.h"

namespace hd {
namespace detail {

template <typename T>
inline void update_maximum(std::atomic<T>& maximum_value,
                           T const& value) noexcept {
  T prev_value = maximum_value;
  while (prev_value < value &&
         !maximum_value.compare_exchange_weak(prev_value, value)) {}
}
}  // namespace detail

template <typename COORD_T, int N_DIMS>
class HausdorffDistanceEarlyBreak : public HausdorffDistance<COORD_T, N_DIMS> {
  using base_t = HausdorffDistance<COORD_T, N_DIMS>;
  using coord_t = COORD_T;
  using point_t = typename base_t::point_t;

 public:
  struct Config {
    int seed;
    uint32_t n_threads;
  };

  HausdorffDistanceEarlyBreak() = default;

  HausdorffDistanceEarlyBreak(const Config& config) {
    CHECK_GT(config.n_threads, 0);
    config_ = config;
  }

  coord_t CalculateDistance(std::vector<point_t>& points_a,
                            std::vector<point_t>& points_b) override {
    std::mt19937 g(config_.seed);  // Mersenne Twister engine seeded with rd()
    std::shuffle(points_a.begin(), points_a.end(), g);
    std::shuffle(points_b.begin(), points_b.end(), g);

    std::vector<std::thread> threads;
    auto thread_count = config_.n_threads;
    auto avg_points = (points_a.size() + thread_count - 1) / thread_count;
    std::atomic<coord_t> cmax;
    std::atomic_uint64_t compared_pairs = 0;

    cmax = 0;

    for (int tid = 0; tid < thread_count; tid++) {
      threads.emplace_back(std::thread([&, tid]() {
        auto begin = tid * avg_points;
        auto end = std::min(begin + avg_points, points_a.size());
        uint64_t local_compared_pairs = 0;

        for (int i = begin; i < end; i++) {
          auto cmin = std::numeric_limits<coord_t>::max();
          for (size_t j = 0; j < points_b.size(); j++) {
            auto d = EuclideanDistance2(points_a[i], points_b[j]);
            local_compared_pairs++;
            if (d < cmin) {
              cmin = d;
            }
            if (cmin < cmax) {
              break;
            }
          }
          if (cmin != std::numeric_limits<coord_t>::max()) {
            detail::update_maximum(cmax, cmin);
          }
        }
        compared_pairs += local_compared_pairs;
      }));
    }

    for (auto& thread : threads) {
      thread.join();
    }

    auto& stats = this->stats_;

    stats["Seed"] = config_.seed;
    stats["Algorithm"] = "Early Break";
    stats["Execution"] = "CPU";
    stats["Threads"] = thread_count;
    stats["ComparedPairs"] = compared_pairs.load();

    return sqrt(cmax);
  }

  coord_t CalculateDistance(const Stream& stream,
                            thrust::device_vector<point_t>& points_a,
                            thrust::device_vector<point_t>& points_b) override {
    thrust::default_random_engine g(config_.seed);
    SharedValue<coord_t> cmax;
    SharedValue<uint32_t> compared_pairs;
    auto* p_cmax = cmax.data();
    auto* p_compared_pairs = compared_pairs.data();
    ArrayView<point_t> v_points_a(points_a);
    ArrayView<point_t> v_points_b(points_b);

    cmax.set(stream.cuda_stream(), 0);
    compared_pairs.set(stream.cuda_stream(), 0);

    thrust::shuffle(thrust::cuda::par.on(stream.cuda_stream()),
                    points_a.begin(), points_a.end(), g);

    thrust::shuffle(thrust::cuda::par.on(stream.cuda_stream()),
                    points_b.begin(), points_b.end(), g);

    LaunchKernel(stream, [=] __device__() {
      using BlockReduce = cub::BlockReduce<coord_t, MAX_BLOCK_SIZE>;
      __shared__ typename BlockReduce::TempStorage temp_storage;
      __shared__ bool early_break;
      __shared__ const point_t* point_a;

      for (auto i = blockIdx.x; i < v_points_a.size(); i += gridDim.x) {
        auto size_b_roundup =
            div_round_up(v_points_b.size(), blockDim.x) * blockDim.x;
        coord_t cmin = std::numeric_limits<coord_t>::max();
        uint32_t n_pairs = 0;

        if (threadIdx.x == 0) {
          early_break = false;
          point_a = &v_points_a[i];
        }
        __syncthreads();

        for (auto j = threadIdx.x; j < size_b_roundup && !early_break;
             j += blockDim.x) {
          auto d = std::numeric_limits<coord_t>::max();
          if (j < v_points_b.size()) {
            const auto& point_b = v_points_b[j];
            d = EuclideanDistance2(*point_a, point_b);
            n_pairs++;
          }

          auto agg_min = BlockReduce(temp_storage).Reduce(d, cub::Min());

          if (threadIdx.x == 0) {
            cmin = std::min(cmin, agg_min);
            if (cmin <= *p_cmax) {
              early_break = true;
            }
          }
          __syncthreads();
        }

        atomicAdd(p_compared_pairs, n_pairs);
        __syncthreads();
        if (threadIdx.x == 0 && cmin != std::numeric_limits<coord_t>::max()) {
          atomicMax(p_cmax, cmin);
        }
      }
    });
    auto n_compared_pairs = compared_pairs.get(stream.cuda_stream());

    auto& stats = this->stats_;

    stats["Seed"] = config_.seed;
    stats["Algorithm"] = "Early Break";
    stats["Execution"] = "GPU";
    stats["ComparedPairs"] = n_compared_pairs;

    return sqrt(cmax.get(stream.cuda_stream()));
  }

 private:
  Config config_;
};
}  // namespace hd

#endif  // HAUSDORFF_DISTANCE_EARLY_BREAK_H
