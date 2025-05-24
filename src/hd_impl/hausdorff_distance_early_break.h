#ifndef HAUSDORFF_DISTANCE_EARLY_BREAK_H
#define HAUSDORFF_DISTANCE_EARLY_BREAK_H
#include <glog/logging.h>
#include <hdr/hdr_histogram.h>
#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <thrust/shuffle.h>

#include <atomic>
#include <thread>
#include <vector>

#include "geoms/distance.h"
#include "hausdorff_distance.h"
#include "running_stats.h"
#include "utils/array_view.h"
#include "utils/derived_atomic_functions.h"
#include "utils/shared_value.h"
#include "utils/stopwatch.h"
#include "utils/stream.h"
#include "utils/type_traits.h"

namespace hd {

template <typename COORD_T, int N_DIMS>
class HausdorffDistanceEarlyBreak : public HausdorffDistance<COORD_T, N_DIMS> {
  using base_t = HausdorffDistance<COORD_T, N_DIMS>;
  using coord_t = COORD_T;
  using point_t = typename base_t::point_t;

 public:
  struct Config {
    int seed;
    uint32_t n_threads = 1;
  };

  HausdorffDistanceEarlyBreak() = default;

  HausdorffDistanceEarlyBreak(const Config& config) {
    CHECK_GT(config.n_threads, 0);
    config_ = config;
  }

  coord_t CalculateDistance(std::vector<point_t>& points_a,
                            std::vector<point_t>& points_b) override {
    double shuffle_time, compute_time;

    Stopwatch sw;
    sw.start();
    std::mt19937 g(config_.seed);  // Mersenne Twister engine seeded with rd()
    std::shuffle(points_a.begin(), points_a.end(), g);
    std::shuffle(points_b.begin(), points_b.end(), g);
    sw.stop();
    shuffle_time = sw.ms();

    sw.start();

    auto thread_count = config_.n_threads;
    auto avg_points = (points_a.size() + thread_count - 1) / thread_count;
    std::atomic<coord_t> cmax2 = 0;
    std::atomic_uint64_t compared_points = 0;

    auto compute = [&](int tid) {
      auto begin = tid * avg_points;
      auto end = std::min(begin + avg_points, points_a.size());
      uint64_t local_compared_pairs = 0;

      for (int i = begin; i < end; i++) {
        auto cmin2 = std::numeric_limits<coord_t>::max();
        for (size_t j = 0; j < points_b.size(); j++) {
          auto d = EuclideanDistance2(points_a[i], points_b[j]);
          local_compared_pairs++;
          if (d < cmin2) {
            cmin2 = d;
          }
          if (cmin2 < cmax2) {
            break;
          }
        }
        if (cmin2 != std::numeric_limits<coord_t>::max()) {
          update_maximum(cmax2, cmin2);
        }
      }
      compared_points += local_compared_pairs;
    };

    if (thread_count == 1) {
      compute(0);
    } else {
      std::vector<std::thread> threads;

      for (int tid = 0; tid < thread_count; tid++) {
        threads.emplace_back(std::thread(compute, tid));
      }

      for (auto& thread : threads) {
        thread.join();
      }
    }

    sw.stop();
    compute_time = sw.ms();

    auto& stats = this->stats_;

    stats["Algorithm"] = "Early Break";
    stats["Execution"] = "CPU";
    stats["Threads"] = thread_count;
    stats["ComparedPairs"] = compared_points.load();
    stats["ShuffleTime"] = shuffle_time;
    stats["ComputeTime"] = compute_time;
    stats["ReportedTime"] = compute_time;

    return sqrt(cmax2);
  }

  coord_t CalculateDistance(const Stream& stream,
                            thrust::device_vector<point_t>& points_a,
                            thrust::device_vector<point_t>& points_b) override {
    Stopwatch sw;
    auto& stats = this->stats_;

    sw.start();
    thrust::default_random_engine g(config_.seed);
    SharedValue<coord_t> cmax2;
    auto* p_cmax2 = cmax2.data();
    ArrayView<point_t> v_points_a(points_a);
    ArrayView<point_t> v_points_b(points_b);

    cmax2.set(stream.cuda_stream(), 0);

    thrust::shuffle(thrust::cuda::par.on(stream.cuda_stream()),
                    points_a.begin(), points_a.end(), g);

    thrust::shuffle(thrust::cuda::par.on(stream.cuda_stream()),
                    points_b.begin(), points_b.end(), g);

    uint32_t* p_point_counters = nullptr;

#ifdef PROFILING
    point_counters_.resize(points_a.size());
    p_point_counters = thrust::raw_pointer_cast(point_counters_.data());
    stats["Profiling"] = true;
#else
    stats["Profiling"] = false;
#endif


    LaunchKernel(stream, [=] __device__ () {
      using BlockReduce = cub::BlockReduce<coord_t, MAX_BLOCK_SIZE>;
      __shared__ typename BlockReduce::TempStorage temp_storage;
      __shared__ bool early_break;
      __shared__ point_t point_a;

      for (auto i = blockIdx.x; i < v_points_a.size(); i += gridDim.x) {
        auto size_b_roundup =
            div_round_up(v_points_b.size(), blockDim.x) * blockDim.x;
        if (threadIdx.x == 0) {
          early_break = false;
          point_a = v_points_a[i];
        }

        __syncthreads();
        int n_iter = 0;
        auto thread_min = std::numeric_limits<coord_t>::max();
        uint32_t n_pairs = 0;

        for (auto j = threadIdx.x; j < size_b_roundup && !early_break;
             j += blockDim.x, n_iter++) {
          if (j < v_points_b.size()) {
            const auto& point_b = v_points_b[j];
            thread_min =
                std::min(thread_min, EuclideanDistance2(point_a, point_b));
            n_pairs++;
          }

          // Reduce the frequency of sync
          if (n_iter % 32 == 0) {
            auto agg_min =
                BlockReduce(temp_storage).Reduce(thread_min, cub::Min());

            if (threadIdx.x == 0) {
              if (agg_min <= *p_cmax2) {
                early_break = true;
              }
            }
            __syncthreads();
          }
        }
        auto agg_cmin2 =
            BlockReduce(temp_storage).Reduce(thread_min, cub::Min());
#ifdef PROFILING
        auto agg_pairs = BlockReduce(temp_storage).Reduce(n_pairs, cub::Sum());
#endif

        if (threadIdx.x == 0) {
          if (!early_break &&
              agg_cmin2 != std::numeric_limits<coord_t>::max()) {
            atomicMax(p_cmax2, agg_cmin2);
          }
#ifdef PROFILING
          p_point_counters[i] = agg_pairs;
#endif
        }
      }
    });

    stream.Sync();
    sw.stop();

    stats["Algorithm"] = "Early Break";
    stats["Execution"] = "GPU";
    stats["ReportedTime"] = sw.ms();

    return sqrt(cmax2.get(stream.cuda_stream()));
  }

  const thrust::device_vector<uint32_t>& get_point_counters() const {
    return point_counters_;
  }

 private:
  Config config_;
  thrust::device_vector<uint32_t> point_counters_;

  template <typename T>
  void update_maximum(std::atomic<T>& maximum_value, T const& value) noexcept {
    T prev_value = maximum_value;
    while (prev_value < value &&
           !maximum_value.compare_exchange_weak(prev_value, value)) {}
  }
};
}  // namespace hd

#endif  // HAUSDORFF_DISTANCE_EARLY_BREAK_H
