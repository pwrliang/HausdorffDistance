#ifndef HAUSDORFF_DISTANCE_RT_HDIST_H
#define HAUSDORFF_DISTANCE_RT_HDIST_H
#include <glog/logging.h>
#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <thrust/shuffle.h>

#include <sstream>

#include "geoms/distance.h"
#include "geoms/hd_bounds.h"
#include "geoms/mbr.h"
#include "hausdorff_distance.h"
#include "hd_impl/primitive_utils.h"
#include "index/quantized_grid.h"
#include "rt/launch_parameters.h"
#include "rt/reusable_buffer.h"
#include "rt/rt_engine.h"
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
class HausdorffDistanceRTHDist : public HausdorffDistance<COORD_T, N_DIMS> {
  using base_t = HausdorffDistance<COORD_T, N_DIMS>;
  using coord_t = COORD_T;
  using point_t = typename base_t::point_t;
  using mbr_t = Mbr<COORD_T, N_DIMS>;
  using grid_t = QuantizedGrid<COORD_T, N_DIMS>;
  using quantized_point_t = typename grid_t::quantized_point_t;
  using query_t =
      typename details::LaunchParamsNNQuantizedGrid<COORD_T, N_DIMS>::Query;

 public:
  struct Config {
    const char* ptx_root;
    bool fast_build = false;
    bool compact = false;
    bool rebuild_bvh = false;
    int bit_count = 7;  // 6-8 according to the paper
    int max_reg_count = 0;
  };

  HausdorffDistanceRTHDist() = default;

  explicit HausdorffDistanceRTHDist(const Config& config) : config_(config) {
    auto rt_config = details::get_default_rt_config(config_.ptx_root);

    rt_config.max_reg_count = config_.max_reg_count;
    rt_engine_.Init(rt_config);
  }

  coord_t CalculateDistance(const Stream& stream,
                            thrust::device_vector<point_t>& points_a,
                            thrust::device_vector<point_t>& points_b) override {
    Stopwatch sw,sw_total;

    sw_total.start();
    uint64_t compared_points = 0;
    auto& stats = this->stats_;
    auto n_points_a = points_a.size();
    auto n_points_b = points_b.size();
    auto center_point_a = CalculateCenterPoint(stream, points_a);
    auto center_point_b = CalculateCenterPoint(stream, points_b);

    // according to the paper
    auto half_dist =
        sqrt(EuclideanDistance2(center_point_a, center_point_b)) / 2;

    cmax2_.set(stream.cuda_stream(), 0);

    stats.clear();

    stats["FastBuildBVH"] = config_.fast_build;
    stats["RebuildBVH"] = config_.rebuild_bvh;


    in_queue_.Init(n_points_a);
    hit_queue_.Init(n_points_a);
    miss_queue_.Init(n_points_a);
    in_queue_.Clear(stream.cuda_stream());
    hit_queue_.Clear(stream.cuda_stream());
    miss_queue_.Clear(stream.cuda_stream());

    OptixTraversableHandle gas_handle;

    // Build BVH
    buffer_.Clear();
    const auto mbr_b = CalculateMbr(stream, points_b.begin(), points_b.end());
    grid_ = grid_t(mbr_b, config_.bit_count);
    grid_.Insert(stream, points_b);

    auto representative_points = grid_.GetRepresentativePoints(stream);

    auto n_diagonals = ceil(half_dist / grid_.GetDiagonalLength());
    COORD_T radius = n_diagonals * grid_.GetDiagonalQuantizedLength();

    VLOG(1) << "half_dist " << half_dist << " radius " << radius
            << " n_diagonals " << n_diagonals << " cell diagonal "
            << grid_.GetDiagonalLength();

    stats["InitRadius"] = radius;

    sw.start();
    auto mem_bytes = rt_engine_.EstimateMemoryUsageForAABB(
        representative_points.size(), config_.fast_build, config_.compact);
    buffer_.Init(mem_bytes * 1.2);
    gas_handle = BuildBVH(stream, representative_points, radius);
    stream.Sync();
    sw.stop();

    stats["BVHBuildTime"] = sw.ms();
    stats["BVHMemoryKB"] = mem_bytes / 1024;
    int iter = 0;
    uint32_t in_size = n_points_a;
    auto d_in_queue = in_queue_.DeviceObject();

    thrust::for_each(thrust::cuda::par.on(stream.cuda_stream()),
                     thrust::make_counting_iterator<uint32_t>(0),
                     thrust::make_counting_iterator<uint32_t>(in_size),
                     [=] __device__(uint32_t i) mutable {
                       d_in_queue.Append(query_t(i, 0));
                     });

    while (in_size > 0) {
      iter++;
      auto& json_iter = stats["Iter" + std::to_string(iter)];

      sw.start();
      details::LaunchParamsNNQuantizedGrid<COORD_T, N_DIMS> params;

      miss_queue_.Clear(stream.cuda_stream());

      params.in_queue = ArrayView<query_t>(in_queue_.data(), in_size);
      params.hit_queue = hit_queue_.DeviceObject();
      params.miss_queue = miss_queue_.DeviceObject();
      params.points_a = points_a;
      params.points_b = points_b;
      params.representative_points = representative_points;
      params.handle = gas_handle;
      params.cmax2 = cmax2_.data();
      params.radius = radius;
      params.prefix_sum = grid_.get_prefix_sum();
      params.point_b_ids = grid_.get_point_ids();
      params.grid = grid_.DeviceObject();

      rt_engine_.CopyLaunchParams(stream.cuda_stream(), params);
      rt_engine_.Render(stream.cuda_stream(), getRTModule(),
                        dim3{in_size, 1, 1});
      auto cmax2 = cmax2_.get(stream.cuda_stream());
      sw.stop();
      auto hit_size = hit_queue_.size(stream.cuda_stream());
      auto miss_size = miss_queue_.size(stream.cuda_stream());

      auto cmax = sqrt(cmax2);

      VLOG(1) << "Iter: " << iter << " radius: " << radius << std::fixed
              << std::setprecision(8) << " cmax2: " << cmax2
              << " cmax: " << cmax << " in_size: " << in_size
              << " hit_size: " << hit_queue_.size(stream.cuda_stream())
              << " miss_size: " << miss_queue_.size(stream.cuda_stream())
              << " Time: " << sw.ms() << " ms";

      in_queue_.Clear(stream.cuda_stream());
      in_queue_.Swap(hit_queue_);
      d_in_queue = in_queue_.DeviceObject();  // update the current dev obj
      auto n_miss = miss_queue_.size(stream.cuda_stream());

      thrust::for_each(thrust::cuda::par.on(stream.cuda_stream()),
                       miss_queue_.data(), miss_queue_.data() + n_miss,
                       [=] __device__(uint32_t point_id) mutable {
                         d_in_queue.Append(query_t(point_id, 0));
                       });

      json_iter["NumInputPoints"] = in_size;
      in_size = in_queue_.size(stream.cuda_stream());
      json_iter["NumOutputPoints"] = in_size;
      json_iter["CMax2"] = cmax2;
      json_iter["RTTime"] = sw.ms();
      json_iter["Radius"] = radius;

      if (in_size > 0) {
        // if (hit_size == 0) {
          // radius *= 2;
        // } else {
        radius += grid_.GetDiagonalQuantizedLength();
        // }
        sw.start();
        if (config_.rebuild_bvh) {
          buffer_.Clear();
          gas_handle = BuildBVH(stream, representative_points, radius);
        } else {
          gas_handle =
              UpdateBVH(stream, gas_handle, representative_points, radius);
        }
        stream.Sync();
        sw.stop();
        json_iter["AdjustBVHTime"] = sw.ms();
      }
    }
    auto cmax2 = cmax2_.get(stream.cuda_stream());
    sw_total.stop();

    stats["Algorithm"] = "RT-HDIST";
    stats["Execution"] = "GPU";
    stats["ComparedPairs"] = compared_points;
    stats["ReportedTime"] = sw_total.ms();

    return sqrt(cmax2);
  }

  OptixTraversableHandle BuildBVH(const Stream& stream,
                                  ArrayView<quantized_point_t> points,
                                  COORD_T radius) {
    aabbs_.resize(points.size());
    thrust::transform(thrust::cuda::par.on(stream.cuda_stream()),
                      points.begin(), points.end(), aabbs_.begin(),
                      [=] __device__(const quantized_point_t& p) {
                        return details::GetOptixAABB(p, radius);
                      });
    return rt_engine_.BuildAccelCustom(stream.cuda_stream(),
                                       ArrayView<OptixAabb>(aabbs_), buffer_,
                                       config_.fast_build, config_.compact);
  }

  OptixTraversableHandle UpdateBVH(const Stream& stream,
                                   OptixTraversableHandle handle,
                                   ArrayView<quantized_point_t> points,
                                   COORD_T radius) {
    aabbs_.resize(points.size());
    thrust::transform(thrust::cuda::par.on(stream.cuda_stream()),
                      points.begin(), points.end(), aabbs_.begin(),
                      [=] __device__(const quantized_point_t& p) {
                        return details::GetOptixAABB(p, radius);
                      });
    return rt_engine_.UpdateAccelCustom(stream.cuda_stream(), handle,
                                        ArrayView<OptixAabb>(aabbs_), buffer_,
                                        0, config_.fast_build, config_.compact);
  }

 private:
  Config config_;
  grid_t grid_;
  thrust::device_vector<OptixAabb> aabbs_;
  SharedValue<mbr_t> mbr_;
  Queue<query_t> in_queue_, hit_queue_;
  Queue<uint32_t> miss_queue_;
  ReusableBuffer buffer_;
  details::RTEngine rt_engine_;
  SharedValue<COORD_T> cmax2_;

  details::ModuleIdentifier getRTModule() {
    details::ModuleIdentifier mod_nn = details::NUM_MODULE_IDENTIFIERS;

    if (typeid(COORD_T) == typeid(float)) {
      if (N_DIMS == 2) {
        mod_nn = details::MODULE_ID_FLOAT_NN_QUANTIZED_GRID_2D;
      } else if (N_DIMS == 3) {
        mod_nn = details::MODULE_ID_FLOAT_NN_QUANTIZED_GRID_3D;
      }
    } else if (typeid(COORD_T) == typeid(double)) {
      if (N_DIMS == 2) {
        mod_nn = details::MODULE_ID_DOUBLE_NN_QUANTIZED_GRID_2D;
      } else if (N_DIMS == 3) {
        mod_nn = details::MODULE_ID_DOUBLE_NN_QUANTIZED_GRID_3D;
      }
    }
    return mod_nn;
  }
};
}  // namespace hd

#endif  //  HAUSDORFF_DISTANCE_RT_HDIST_H
