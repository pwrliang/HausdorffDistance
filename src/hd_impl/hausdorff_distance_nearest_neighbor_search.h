#ifndef HAUSDORFF_DISTANCE_NEAREST_NEIGHBOR_SEARCH_H
#define HAUSDORFF_DISTANCE_NEAREST_NEIGHBOR_SEARCH_H
#include <glog/logging.h>
#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <thrust/shuffle.h>

#include <sstream>

#include "cukd/builder.h"
#include "cukd/fcp.h"
#include "geoms/distance.h"
#include "geoms/mbr.h"
#include "hausdorff_distance.h"
#include "utils/array_view.h"
#include "utils/derived_atomic_functions.h"
#include "utils/helpers.h"
#include "utils/shared_value.h"
#include "utils/stopwatch.h"
#include "utils/stream.h"
#include "utils/type_traits.h"
#include "utils/util.h"

namespace hd {

template <typename COORD_T, int N_DIMS>
class HausdorffDistanceNearestNeighborSearch
    : public HausdorffDistance<COORD_T, N_DIMS> {
  using base_t = HausdorffDistance<COORD_T, N_DIMS>;
  using coord_t = COORD_T;
  using point_t = typename base_t::point_t;
  using data_traits = cukd::default_data_traits<point_t>;
  using kdtree_t = cukd::SpatialKDTree<point_t, data_traits>;

 public:
  struct Config {
    uint32_t n_threads = 1;
  };

  HausdorffDistanceNearestNeighborSearch() = default;

  HausdorffDistanceNearestNeighborSearch(const Config& config) {
    CHECK_GT(config.n_threads, 0);
    config_ = config;
  }

  COORD_T CalculateDistance(std::vector<point_t>& points_a,
                            std::vector<point_t>& points_b) override {
    double build_time, compute_time;
    Stopwatch sw;
    Stream stream;

    sw.start();
    thrust::device_vector<point_t> temp_points = points_b;
    kd_tree_b_.Build(stream, temp_points);
    sw.stop();

    build_time = sw.ms();

    sw.start();
    std::atomic<coord_t> cmax2 = 0;

    auto thread_count = config_.n_threads;
    auto avg_points = (points_a.size() + thread_count - 1) / thread_count;

    auto compute = [&](int tid) {
      auto begin = tid * avg_points;
      auto end = std::min(begin + avg_points, points_a.size());
      COORD_T local_cmax2 = 0;

      for (int i = begin; i < end; i++) {
        const auto& p_a = points_a[i];
        auto point_b_id = kd_tree_b_.fcp(p_a);
        auto dist2 = EuclideanDistance2(p_a, points_b[point_b_id]);
        local_cmax2 = std::max(local_cmax2, dist2);
      }
      update_maximum(cmax2, local_cmax2);
    };

    if (thread_count == 1) {
      compute(0);
    } else {
      std::vector<std::thread> threads;
      for (int tid = 0; tid < thread_count; tid++) {
        threads.emplace_back(compute, tid);
      }

      for (auto& thread : threads) {
        thread.join();
      }
    }

    sw.stop();

    compute_time = sw.ms();

    auto& stats = this->stats_;

    stats["Algorithm"] = "Nearest Neighbor Search";
    stats["Execution"] = "CPU";
    stats["Threads"] = thread_count;
    stats["ComparedPairs"] = points_a.size();
    stats["BuildIndexTime"] = build_time;
    stats["ComputeTime"] = compute_time;
    stats["ReportedTime"] = build_time + compute_time;

    return sqrt(cmax2);
  }

  coord_t CalculateDistance(const Stream& stream,
                            thrust::device_vector<point_t>& points_a,
                            thrust::device_vector<point_t>& points_b) override {
    double build_time, compute_time;

    kdtree_t tree_b;
    ArrayView<point_t> v_points_a(points_a);
    ArrayView<point_t> v_points_b(points_b);
    Stopwatch sw;
    auto& stats = this->stats_;

    sw.start();
    cukd::box_t<point_t>* d_bounds;
    cudaMallocManaged((void**) &d_bounds, sizeof(cukd::box_t<point_t>));

    cukd::buildTree<point_t, data_traits>(v_points_b.data(), v_points_b.size(),
                                          d_bounds, stream.cuda_stream());

    stream.Sync();
    sw.stop();
    build_time = sw.ms();

    sw.start();
    auto max2 = thrust::transform_reduce(
        thrust::cuda::par.on(stream.cuda_stream()), points_a.begin(),
        points_a.end(),
        [=] __device__(const point_t& p) mutable {
          auto point_b_id = cukd::cct::fcp<point_t, data_traits>(
              p, *d_bounds, v_points_b.data(), v_points_b.size());
          return EuclideanDistance2(p, v_points_b[point_b_id]);
        },
        0, thrust::maximum<coord_t>());

    cudaFreeAsync(d_bounds, stream.cuda_stream());
    stream.Sync();
    sw.stop();
    compute_time = sw.ms();

    stats["Algorithm"] = "Nearest Neighbor Search";
    stats["Execution"] = "GPU";
    stats["BuildIndexTime"] = build_time;
    stats["ComputeTime"] = compute_time;
    stats["ReportedTime"] = build_time + compute_time;

    return sqrt(max2);
  }

 private:
  Config config_;
  KDTree<COORD_T, N_DIMS> kd_tree_b_;
  template <typename T>
  void update_maximum(std::atomic<T>& maximum_value, T const& value) noexcept {
    T prev_value = maximum_value;
    while (prev_value < value &&
           !maximum_value.compare_exchange_weak(prev_value, value)) {}
  }
};
}  // namespace hd

#endif  // HAUSDORFF_DISTANCE_NEAREST_NEIGHBOR_SEARCH_H
