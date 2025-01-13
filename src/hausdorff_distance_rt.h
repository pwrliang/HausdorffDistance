#ifndef HAUSDORFF_DISTANCE_RT_H
#define HAUSDORFF_DISTANCE_RT_H
#include <flags.h>
#include <glog/logging.h>
#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <thrust/shuffle.h>

#include "distance.h"
#include "rt/reusable_buffer.h"
#include "rt/rt_engine.h"
#include "sampler.h"
#include "utils/derived_atomic_functions.h"
#include "utils/helpers.h"
#include "utils/queue.h"
#include "utils/stopwatch.h"
#include "utils/stream.h"
#include "utils/type_traits.h"
#include "utils/util.h"

namespace hd {

namespace details {

DEV_HOST_INLINE OptixAabb GetOptixAABB(float2 p, float radius) {
  OptixAabb aabb;
  aabb.minX = p.x - radius;
  aabb.maxX = p.x + radius;
  aabb.minY = p.y - radius;
  aabb.maxY = p.y + radius;
  aabb.minZ = aabb.maxZ = 0;
  return aabb;
}

DEV_HOST_INLINE OptixAabb GetOptixAABB(float3 p, float radius) {
  OptixAabb aabb;
  aabb.minX = p.x - radius;
  aabb.maxX = p.x + radius;
  aabb.minY = p.y - radius;
  aabb.maxY = p.y + radius;
  aabb.minZ = p.z - radius;
  aabb.maxZ = p.z + radius;
  return aabb;
}

DEV_HOST_INLINE OptixAabb GetOptixAABB(double2 p, double radius) {
  OptixAabb aabb;
  aabb.minX = next_float_from_double(p.x - radius, -1, 2);
  aabb.maxX = next_float_from_double(p.x + radius, 1, 2);
  aabb.minY = next_float_from_double(p.y - radius, -1, 2);
  aabb.maxY = next_float_from_double(p.y + radius, 1, 2);
  aabb.minZ = aabb.maxZ = 0;
  return aabb;
}

DEV_HOST_INLINE OptixAabb GetOptixAABB(double3 p, double radius) {
  OptixAabb aabb;
  aabb.minX = next_float_from_double(p.x - radius, -1, 2);
  aabb.maxX = next_float_from_double(p.x + radius, 1, 2);
  aabb.minY = next_float_from_double(p.y - radius, -1, 2);
  aabb.maxY = next_float_from_double(p.y + radius, 1, 2);
  aabb.minZ = next_float_from_double(p.z - radius, -1, 2);
  aabb.maxZ = next_float_from_double(p.z + radius, 1, 2);
  return aabb;
}

}  // namespace details

struct HausdorffDistanceRTConfig {
  const char* ptx_root;
  bool fast_build = false;
  bool compact = false;
  float sample_rate = 0.1;
};

template <typename COORD_T, int N_DIMS>
class HausdorffDistanceRT {
  using coord_t = COORD_T;
  using point_t = typename cuda_vec<COORD_T, N_DIMS>::type;
  using mbr_t = Mbr<COORD_T, N_DIMS>;

 public:
  HausdorffDistanceRT() = default;

  void Init(const HausdorffDistanceRTConfig& hd_config) {
    config_ = hd_config;
    auto rt_config = details::get_default_rt_config(hd_config.ptx_root);
    rt_engine_.Init(rt_config);
  }

  template <typename IT>
  void SetPointsTo(const Stream& stream, IT begin, IT end) {
    auto* p_mbr = mbr_b_.data();

    points_b_.assign(begin, end);

    thrust::shuffle(thrust::cuda::par.on(stream.cuda_stream()),
                    points_b_.begin(), points_b_.end(), g_);

    mbr_b_.set(stream.cuda_stream(), mbr_t());
    thrust::for_each(
        thrust::cuda::par.on(stream.cuda_stream()), points_b_.begin(),
        points_b_.end(),
        [=] __device__(const point_t& p) mutable { p_mbr->Expand(p); });
    auto mbr_b = mbr_b_.get(stream.cuda_stream());

    auto n_points = points_b_.size();
    auto mem_bytes = rt_engine_.EstimateMemoryUsageForAABB(
        n_points, config_.fast_build, config_.compact);

    buffer_.Init(mem_bytes * 1.5);
    buffer_.Clear();
  }

  template <typename IT>
  COORD_T CalculateDistanceFrom(const Stream& stream, IT begin, IT end) {
    auto* p_mbr = mbr_a_.data();

    points_a_.assign(begin, end);

    thrust::shuffle(thrust::cuda::par.on(stream.cuda_stream()),
                    points_a_.begin(), points_a_.end(), g_);

    mbr_a_.set(stream.cuda_stream(), mbr_t());
    thrust::for_each(
        thrust::cuda::par.on(stream.cuda_stream()), points_a_.begin(),
        points_a_.end(),
        [=] __device__(const point_t& p) mutable { p_mbr->Expand(p); });
    auto mbr_a = mbr_a_.get(stream.cuda_stream());
    auto mbr_b = mbr_b_.get(stream.cuda_stream());
    HdBounds<COORD_T, N_DIMS> hd_bounds(mbr_a);
    auto hd_lb = hd_bounds.GetLowerBound(mbr_b);
    auto hd_ub = hd_bounds.GetUpperBound(mbr_b);
    COORD_T radius = hd_lb;

    in_queue_.Init(points_a_.size());
    out_queue_.Init(points_a_.size());

    in_queue_.Clear(stream.cuda_stream());
    out_queue_.Clear(stream.cuda_stream());

    auto d_in_queue = in_queue_.DeviceObject();

    thrust::for_each(thrust::cuda::par.on(stream.cuda_stream()),
                     thrust::make_counting_iterator<uint32_t>(0),
                     thrust::make_counting_iterator<uint32_t>(points_a_.size()),
                     [=] __device__(uint32_t point_id) mutable {
                       d_in_queue.Append(point_id);
                     });

    uint32_t in_size = points_a_.size();
    LOG(INFO) << "LB " << hd_lb << " UB " << hd_ub;

    buffer_.Clear();
    auto gas_handle = BuildBVH(stream, points_b_, radius);

    SharedValue<uint32_t> skip_count;
    SharedValue<uint32_t> skip_total_idx;
    SharedValue<uint32_t> iter_hits;
    int iter = 0;
    uint64_t total_hits = 0;

    cmax2_.set(stream.cuda_stream(), hd_lb * hd_lb);

    while (in_size > 0) {
      iter++;
      iter_hits.set(stream.cuda_stream(), 0);
      details::LaunchParamsNN<COORD_T, N_DIMS> params_nn;

      skip_count.set(stream.cuda_stream(), 0);
      skip_total_idx.set(stream.cuda_stream(), 0);
      ArrayView<uint32_t> in(in_queue_.data(), in_size);

      params_nn.in_queue = in;
      params_nn.out_queue = out_queue_.DeviceObject();
      params_nn.points_a = ArrayView<point_t>(points_a_);
      params_nn.points_b = ArrayView<point_t>(points_b_);
      params_nn.handle = gas_handle;
      params_nn.aabbs = ArrayView<OptixAabb>(aabbs_);
      params_nn.cmax2 = cmax2_.data();
      params_nn.radius = radius;
      params_nn.skip_count = skip_count.data();
      params_nn.skip_total_idx = skip_total_idx.data();
      params_nn.n_hits = iter_hits.data();

      details::ModuleIdentifier mod_nn = details::NUM_MODULE_IDENTIFIERS;

      if (typeid(COORD_T) == typeid(float)) {
        if (N_DIMS == 2) {
          mod_nn = details::MODULE_ID_FLOAT_NN_2D;
        } else if (N_DIMS == 3) {
          mod_nn = details::MODULE_ID_FLOAT_NN_3D;
        }
      } else if (typeid(COORD_T) == typeid(double)) {
        if (N_DIMS == 2) {
          mod_nn = details::MODULE_ID_DOUBLE_NN_2D;
        } else if (N_DIMS == 3) {
          mod_nn = details::MODULE_ID_DOUBLE_NN_3D;
        }
      }

      dim3 dims{1, 1, 1};
      Stopwatch sw;
      sw.start();
      dims.x = in_size;
      rt_engine_.CopyLaunchParams(stream.cuda_stream(), params_nn);
      rt_engine_.Render(stream.cuda_stream(), mod_nn, dims);
      stream.Sync();
      sw.stop();

      auto cmax2 = cmax2_.get(stream.cuda_stream());
      auto cmax = sqrt(cmax2);
      auto n_hits = iter_hits.get(stream.cuda_stream());
      total_hits += n_hits;

      VLOG(1) << "Iter: " << iter << " radius: " << radius
              << " cmax2: " << cmax2 << " cmax: " << cmax
              << " in_size: " << in_size
              << " out_size: " << out_queue_.size(stream.cuda_stream())
              << " Avg hits " << (float) n_hits / in_size << " Skip idx: "
              << (float) skip_total_idx.get(stream.cuda_stream()) /
                     skip_count.get(stream.cuda_stream())
              << " Time: " << sw.ms() << " ms";

      in_queue_.Clear(stream.cuda_stream());
      in_queue_.Swap(out_queue_);

      radius *= 1.1;
      in_size = in_queue_.size(stream.cuda_stream());
      if (in_size > 0) {
        gas_handle = UpdateBVH(stream, gas_handle, points_b_, radius);
        // buffer_.Clear();
        // gas_handle = BuildBVH(stream, points_b_, radius);
      }
    }
    LOG(INFO) << "Total Hits " << total_hits;

    return sqrt(cmax2_.get(stream.cuda_stream()));
  }

  OptixTraversableHandle BuildBVH(const Stream& stream,
                                  ArrayView<point_t> points, COORD_T radius) {
    aabbs_.resize(points.size());
    thrust::transform(thrust::cuda::par.on(stream.cuda_stream()),
                      points.begin(), points.end(), aabbs_.begin(),
                      [=] __device__(const point_t& p) {
                        return details::GetOptixAABB(p, radius);
                      });
    return rt_engine_.BuildAccelCustom(stream.cuda_stream(),
                                       ArrayView<OptixAabb>(aabbs_), buffer_,
                                       config_.fast_build, config_.compact);
  }

  OptixTraversableHandle UpdateBVH(const Stream& stream,
                                   OptixTraversableHandle handle,
                                   ArrayView<point_t> points, COORD_T radius) {
    aabbs_.resize(points.size());
    thrust::transform(thrust::cuda::par.on(stream.cuda_stream()),
                      points.begin(), points.end(), aabbs_.begin(),
                      [=] __device__(const point_t& p) {
                        return details::GetOptixAABB(p, radius);
                      });
    return rt_engine_.UpdateAccelCustom(stream.cuda_stream(), handle,
                                        ArrayView<OptixAabb>(aabbs_), buffer_,
                                        0, config_.fast_build, config_.compact);
  }

 private:
  HausdorffDistanceRTConfig config_;
  thrust::default_random_engine g_;
  thrust::device_vector<point_t> points_a_;
  thrust::device_vector<point_t> points_b_;
  thrust::device_vector<OptixAabb> aabbs_;
  SharedValue<mbr_t> mbr_a_;
  SharedValue<mbr_t> mbr_b_;
  OptixTraversableHandle gas_handle_;
  Queue<uint32_t> in_queue_, out_queue_;
  Sampler sampler_;
  ReusableBuffer buffer_;
  details::RTEngine rt_engine_;
  SharedValue<COORD_T> cmax2_;
};
}  // namespace hd

#endif  // HAUSDORFF_DISTANCE_RT_H
