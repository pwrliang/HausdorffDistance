#ifndef HAUSDORFF_DISTANCE_RT_H
#define HAUSDORFF_DISTANCE_RT_H
#include <glog/logging.h>
#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <thrust/shuffle.h>

#include <boost/geometry/algorithms/detail/partition.hpp>
#include <iomanip>

#include "cukd/kdtree.h"
#include "cukd/spatial-kdtree.h"
#include "distance.h"
#include "grid.h"
#include "kdtree/kd_tree_helpers.h"
#include "kdtree/labeled_point.h"
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

#define NEXT_AFTER_ROUNDS (2)

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

DEV_HOST_INLINE OptixAabb GetOptixAABB(float2 p, float radius, int coord_offset,
                                       const Mbr<float, 2>& mbr) {
  OptixAabb aabb;
  aabb.minX = p.x - radius;
  aabb.maxX = p.x + radius;
  aabb.minY = p.y - radius;
  aabb.maxY = p.y + radius;
  aabb.minZ = aabb.maxZ = 0;

  for (int dim = 0; dim < 2; dim++) {
    auto lower = mbr.lower(dim);
    auto upper = mbr.upper(dim);
    auto& min_val = reinterpret_cast<float*>(&aabb.minX)[dim];
    auto& max_val = reinterpret_cast<float*>(&aabb.maxX)[dim];

    assert(min_val >= lower && max_val <= upper);
    auto norm_min_val = (min_val - lower) / (upper - lower);
    auto norm_max_val = (max_val - lower) / (upper - lower);

    assert(norm_min_val >= 0 && norm_min_val <= 1);
    assert(norm_max_val >= 0 && norm_max_val <= 1);

    min_val = norm_min_val + coord_offset;
    max_val = norm_max_val + coord_offset;
  }

  return aabb;
}

DEV_HOST_INLINE OptixAabb GetOptixAABB(float3 p, float radius, int coord_offset,
                                       const Mbr<float, 3>& mbr) {
  OptixAabb aabb;
  aabb.minX = p.x - radius;
  aabb.maxX = p.x + radius;
  aabb.minY = p.y - radius;
  aabb.maxY = p.y + radius;
  aabb.minZ = p.z - radius;
  aabb.maxZ = p.z + radius;

  for (int dim = 0; dim < 3; dim++) {
    auto lower = mbr.lower(dim);
    auto upper = mbr.upper(dim);
    auto& min_val = reinterpret_cast<float*>(&aabb.minX)[dim];
    auto& max_val = reinterpret_cast<float*>(&aabb.maxX)[dim];

    assert(min_val >= lower && max_val <= upper);
    auto norm_min_val = (min_val - lower) / (upper - lower);
    auto norm_max_val = (max_val - lower) / (upper - lower);

    assert(norm_min_val >= 0 && norm_min_val <= 1);
    assert(norm_max_val >= 0 && norm_max_val <= 1);

    min_val = norm_min_val + coord_offset;
    max_val = norm_max_val + coord_offset;
  }

  return aabb;
}

DEV_HOST_INLINE OptixAabb GetOptixAABB(double2 p, double radius,
                                       int coord_offset,
                                       const Mbr<double, 2>& mbr) {
  OptixAabb aabb;
  aabb.minX = next_float_from_double(p.x - radius, -1, 2);
  aabb.maxX = next_float_from_double(p.x + radius, 1, 2);
  aabb.minY = next_float_from_double(p.y - radius, -1, 2);
  aabb.maxY = next_float_from_double(p.y + radius, 1, 2);
  aabb.minZ = aabb.maxZ = 0;

  for (int dim = 0; dim < 2; dim++) {
    auto lower = mbr.lower(dim);
    auto upper = mbr.upper(dim);
    auto& min_val = reinterpret_cast<float*>(&aabb.minX)[dim];
    auto& max_val = reinterpret_cast<float*>(&aabb.maxX)[dim];

    assert(min_val >= lower && max_val <= upper);
    auto norm_min_val = (min_val - lower) / (upper - lower);
    auto norm_max_val = (max_val - lower) / (upper - lower);

    assert(norm_min_val >= 0 && norm_min_val <= 1);
    assert(norm_max_val >= 0 && norm_max_val <= 1);

    min_val = norm_min_val + coord_offset;
    max_val = norm_max_val + coord_offset;
  }

  return aabb;
}

DEV_HOST_INLINE OptixAabb GetOptixAABB(double3 p, double radius,
                                       int coord_offset,
                                       const Mbr<double, 3>& mbr) {
  OptixAabb aabb;
  aabb.minX = next_float_from_double(p.x - radius, -1, 2);
  aabb.maxX = next_float_from_double(p.x + radius, 1, 2);
  aabb.minY = next_float_from_double(p.y - radius, -1, 2);
  aabb.maxY = next_float_from_double(p.y + radius, 1, 2);
  aabb.minZ = next_float_from_double(p.z - radius, -1, 2);
  aabb.maxZ = next_float_from_double(p.z + radius, 1, 2);

  for (int dim = 0; dim < 3; dim++) {
    auto lower = mbr.lower(dim);
    auto upper = mbr.upper(dim);
    auto& min_val = reinterpret_cast<float*>(&aabb.minX)[dim];
    auto& max_val = reinterpret_cast<float*>(&aabb.maxX)[dim];

    assert(min_val >= lower && max_val <= upper);
    auto norm_min_val = (min_val - lower) / (upper - lower);
    auto norm_max_val = (max_val - lower) / (upper - lower);

    assert(norm_min_val >= 0 && norm_min_val <= 1);
    assert(norm_max_val >= 0 && norm_max_val <= 1);

    min_val = norm_min_val + coord_offset;
    max_val = norm_max_val + coord_offset;
  }

  return aabb;
}
template <typename COORD_T, int N_DIMS>
DEV_HOST_INLINE OptixAabb GetOptixAABB(const Mbr<COORD_T, N_DIMS>& mbr,
                                       COORD_T radius) {
  OptixAabb aabb;

  aabb.minZ = aabb.maxZ = 0;

  for (int dim = 0; dim < N_DIMS; dim++) {
    auto lower = mbr.lower(dim);
    auto upper = mbr.upper(dim);
    reinterpret_cast<float*>(&aabb.minX)[dim] = lower - radius;
    reinterpret_cast<float*>(&aabb.maxX)[dim] = upper + radius;
  }
  return aabb;
}
}  // namespace details

struct HausdorffDistanceRTConfig {
  int seed = 0;
  const char* ptx_root;
  bool fast_build = false;
  bool compact = false;
  bool rebuild_bvh = false;
  float radius_step = 2;
  float sample_rate = 0.0001;
  float init_radius = 0;
  int max_samples = 100 * 1000;
  int max_hit = 1000;
};

template <typename COORD_T, int N_DIMS>
class HausdorffDistanceRT {
  using coord_t = COORD_T;
  using point_t = typename cuda_vec<COORD_T, N_DIMS>::type;
  using mbr_t = Mbr<COORD_T, N_DIMS>;
  using labeled_point_t = LabeledPoint<N_DIMS>;

 public:
  HausdorffDistanceRT() = default;

  void Init(const HausdorffDistanceRTConfig& hd_config) {
    config_ = hd_config;
    auto rt_config = details::get_default_rt_config(hd_config.ptx_root);
    rt_engine_.Init(rt_config);
    sampler_.Init(hd_config.max_samples);
    g_ = thrust::default_random_engine(config_.seed);
  }

  // TODO: OptiX 9, using Tensor cores throught RT cores to calculate the
  // distance
  COORD_T CalculateDistance(const Stream& stream,
                            thrust::device_vector<point_t>& points_a,
                            thrust::device_vector<point_t>& points_b) {
    auto n_points_a = points_a.size();
    auto n_points_b = points_b.size();
    const auto mbr_a = CalculateMbr(stream, points_a.begin(), points_a.end());
    const auto mbr_b = CalculateMbr(stream, points_b.begin(), points_b.end());
    auto mem_bytes = rt_engine_.EstimateMemoryUsageForAABB(
        n_points_b, config_.fast_build, config_.compact);

    buffer_.Init(mem_bytes * 1.5);
    buffer_.Clear();

    auto sampled_point_ids_a = sampler_.Sample(
        stream.cuda_stream(), points_a.size(),
        std::max(1u, (uint32_t) (points_a.size() * config_.sample_rate)));

    HdBounds<COORD_T, N_DIMS> hd_bounds(mbr_a);
    auto hd_lb = hd_bounds.GetLowerBound(mbr_b);
    auto hd_ub = hd_bounds.GetUpperBound(mbr_b);

    cmax2_.set(stream.cuda_stream(), hd_lb * hd_lb);

    Stopwatch sw;
    sw.start();
    CalculateHDEarlyBreak(stream, points_a, points_b, sampled_point_ids_a);
    auto sampled_hd2 = cmax2_.get(stream.cuda_stream());
    sw.stop();
    LOG(INFO) << "Sampled HD " << sqrt(sampled_hd2) << " time " << sw.ms();

    COORD_T radius = sqrt(sampled_hd2);

    // radius *= sqrt(N_DIMS);  // Refer RTNN Fig10c
    COORD_T max_radius = hd_ub;

    LOG(INFO) << "LB " << hd_lb << " UB " << hd_ub << " Init Radius " << radius;

    auto world_mbr = mbr_a;
    COORD_T min_extent = std::numeric_limits<COORD_T>::max();
    COORD_T max_extent = 0;
    world_mbr.Expand(mbr_b);

    for (int dim = 0; dim < N_DIMS; dim++) {
      auto lower = world_mbr.lower(dim) - max_radius;
      auto upper = world_mbr.upper(dim) + max_radius;
      world_mbr.set_lower(dim, lower);
      world_mbr.set_upper(dim, upper);
      min_extent = std::min(min_extent, upper - lower);
      max_extent = std::max(max_extent, upper - lower);
    };

    in_queue_.Init(n_points_a);
    term_queue_.Init(n_points_a);
    out_queue_.Init(n_points_a);
    in_queue_.Clear(stream.cuda_stream());
    out_queue_.Clear(stream.cuda_stream());

    auto d_in_queue = in_queue_.DeviceObject();

    thrust::for_each(thrust::cuda::par.on(stream.cuda_stream()),
                     thrust::make_counting_iterator<uint32_t>(0),
                     thrust::make_counting_iterator<uint32_t>(points_a.size()),
                     [=] __device__(uint32_t point_id) mutable {
                       d_in_queue.Append(point_id);
                     });

    uint32_t in_size = n_points_a;

    buffer_.Clear();
    auto gas_handle = BuildBVH(stream, points_b, radius);
    stream.Sync();

    SharedValue<uint32_t> iter_hits;
    int iter = 0;

    while (in_size > 0) {
      iter++;
      iter_hits.set(stream.cuda_stream(), 0);
      details::LaunchParamsNN<COORD_T, N_DIMS> params;
      ArrayView<uint32_t> in(in_queue_.data(), in_size);

      params.in_queue = in;
      params.miss_queue = out_queue_.DeviceObject();
      params.points_a = ArrayView<point_t>(points_a);
      params.points_b = ArrayView<point_t>(points_b);
      params.handle = gas_handle;
      params.cmax2 = cmax2_.data();
      params.radius = radius;
      params.n_hits = iter_hits.data();
      params.max_hit = std::numeric_limits<uint32_t>::max();

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
      rt_engine_.CopyLaunchParams(stream.cuda_stream(), params);
      rt_engine_.Render(stream.cuda_stream(), mod_nn, dims);
      stream.Sync();
      sw.stop();

      auto cmax2 = cmax2_.get(stream.cuda_stream());
      auto cmax = sqrt(cmax2);

      VLOG(1) << "Iter: " << iter << " radius: " << radius << std::fixed
              << std::setprecision(8) << " cmax2: " << cmax2
              << " cmax: " << cmax << " in_size: " << in_size
              << " out_size: " << out_queue_.size(stream.cuda_stream())
              << " Time: " << sw.ms() << " ms";

      in_queue_.Clear(stream.cuda_stream());
      in_queue_.Swap(out_queue_);

      radius *= config_.radius_step;
      in_size = in_queue_.size(stream.cuda_stream());
      if (in_size > 0) {
        if (config_.rebuild_bvh) {
          buffer_.Clear();
          gas_handle = BuildBVH(stream, points_b, radius);
        } else {
          gas_handle = UpdateBVH(stream, gas_handle, points_b, radius);
        }
      }
    }
    return sqrt(cmax2_.get(stream.cuda_stream()));
  }

  COORD_T CalculateDistanceHybrid(const Stream& stream,
                                  thrust::device_vector<point_t>& points_a,
                                  thrust::device_vector<point_t>& points_b) {
    auto n_points_a = points_a.size();
    auto n_points_b = points_b.size();
    const auto mbr_a = CalculateMbr(stream, points_a.begin(), points_a.end());
    const auto mbr_b = CalculateMbr(stream, points_b.begin(), points_b.end());
    auto mem_bytes = rt_engine_.EstimateMemoryUsageForAABB(
        n_points_b, config_.fast_build, config_.compact);

    buffer_.Init(mem_bytes * 1.5);
    buffer_.Clear();

    // Do not shuffle A for better ray-coherence
    thrust::shuffle(thrust::cuda::par.on(stream.cuda_stream()),
                    points_b.begin(), points_b.end(), g_);

    HdBounds<COORD_T, N_DIMS> hd_bounds(mbr_a);
    auto hd_lb = hd_bounds.GetLowerBound(mbr_b);
    auto hd_ub = hd_bounds.GetUpperBound(mbr_b);
    COORD_T radius;

    cmax2_.set(stream.cuda_stream(), hd_lb * hd_lb);

    Stopwatch sw;

    sw.start();
    // TODO: Optimization, brute force if sampled_point_ids_a << points_b
    auto sampled_point_ids_a = sampler_.Sample(
        stream.cuda_stream(), points_a.size(),
        std::max(1u, (uint32_t) (points_a.size() * config_.sample_rate)));
    CalculateHDEarlyBreak(stream, points_a, points_b, sampled_point_ids_a);
    auto sampled_hd2 = cmax2_.get(stream.cuda_stream());
    sw.stop();
    LOG(INFO) << "Sampled HD " << sqrt(sampled_hd2) << " time " << sw.ms();

    // if (hd_lb > 0) {
    //   radius = hd_lb;
    // } else {
    //   radius = hd_ub * 0.01;
    // }
    radius = sqrt(sampled_hd2);
    COORD_T max_radius = hd_ub;

    LOG(INFO) << "LB " << hd_lb << " UB " << hd_ub << " Init Radius " << radius;

    auto world_mbr = mbr_a;
    COORD_T min_extent = std::numeric_limits<COORD_T>::max();
    COORD_T max_extent = 0;
    world_mbr.Expand(mbr_b);

    for (int dim = 0; dim < N_DIMS; dim++) {
      auto lower = world_mbr.lower(dim) - max_radius;
      auto upper = world_mbr.upper(dim) + max_radius;
      world_mbr.set_lower(dim, lower);
      world_mbr.set_upper(dim, upper);
      min_extent = std::min(min_extent, upper - lower);
      max_extent = std::max(max_extent, upper - lower);
    }

    in_queue_.Init(n_points_a);
    term_queue_.Init(n_points_a);
    out_queue_.Init(n_points_a);

    in_queue_.Clear(stream.cuda_stream());
    out_queue_.Clear(stream.cuda_stream());

    auto d_in_queue = in_queue_.DeviceObject();

    thrust::for_each(thrust::cuda::par.on(stream.cuda_stream()),
                     thrust::make_counting_iterator<uint32_t>(0),
                     thrust::make_counting_iterator<uint32_t>(points_a.size()),
                     [=] __device__(uint32_t point_id) mutable {
                       d_in_queue.Append(point_id);
                     });

    uint32_t in_size = n_points_a;

    buffer_.Clear();
    auto gas_handle = BuildBVH(stream, points_b, radius);
    stream.Sync();

    SharedValue<uint32_t> iter_hits;
    SharedValue<uint32_t> n_compared_pairs;
    int iter = 0;
    uint64_t total_hits = 0;
    uint64_t total_compared_pairs = 0;

    while (in_size > 0) {
      iter++;
      iter_hits.set(stream.cuda_stream(), 0);
      n_compared_pairs.set(stream.cuda_stream(), 0);
      term_queue_.Clear(stream.cuda_stream());

      details::LaunchParamsNN<COORD_T, N_DIMS> params;

      params.in_queue = ArrayView<uint32_t>(in_queue_.data(), in_size);
      params.term_queue = term_queue_.DeviceObject();
      params.miss_queue = out_queue_.DeviceObject();
      params.points_a = ArrayView<point_t>(points_a);
      params.points_b = ArrayView<point_t>(points_b);
      params.handle = gas_handle;
      params.cmax2 = cmax2_.data();
      params.radius = radius;
      params.n_hits = iter_hits.data();
      params.max_hit = config_.max_hit;

      details::ModuleIdentifier mod_nn = details::NUM_MODULE_IDENTIFIERS;

      if (typeid(COORD_T) == typeid(float)) {
        if (N_DIMS == 2) {
          mod_nn = details::MODULE_ID_FLOAT_NN_2D;
        } else if (N_DIMS == 3) {
        }
      } else if (typeid(COORD_T) == typeid(double)) {
        if (N_DIMS == 2) {
          mod_nn = details::MODULE_ID_DOUBLE_NN_2D;
        } else if (N_DIMS == 3) {
        }
      }

      dim3 dims{in_size, 1, 1};
      sw.start();
      rt_engine_.CopyLaunchParams(stream.cuda_stream(), params);
      rt_engine_.Render(stream.cuda_stream(), mod_nn, dims);

      ArrayView<uint32_t> term_ids(term_queue_.data(),
                                   term_queue_.size(stream.cuda_stream()));
      sw.stop();
      auto rt_time = sw.ms();

      sw.start();
      if (!term_ids.empty()) {
        auto eb_compared_pairs =
            CalculateHDEarlyBreak(stream, points_a, points_b, term_ids);
      }
      sw.stop();
      auto eb_time = sw.ms();

      LOG(INFO) << "Iter " << iter << " Radius " << radius << " In " << in_size
                << " Term " << term_ids.size() << " Miss "
                << out_queue_.size(stream.cuda_stream()) << " RT Time "
                << rt_time << " ms"
                << " EB Time " << eb_time << " ms";
      auto n_hits = iter_hits.get(stream.cuda_stream());
      auto n_pairs = n_compared_pairs.get(stream.cuda_stream());

      total_hits += n_hits;
      total_compared_pairs += n_pairs;

      in_queue_.Clear(stream.cuda_stream());
      in_queue_.Swap(out_queue_);

      radius *= config_.radius_step;
      in_size = in_queue_.size(stream.cuda_stream());
      if (in_size > 0) {
        if (config_.rebuild_bvh) {
          buffer_.Clear();
          gas_handle = BuildBVH(stream, points_b, radius);
        } else {
          gas_handle = UpdateBVH(stream, gas_handle, points_b, radius);
        }
      }
    }
    LOG(INFO) << "Total Hits " << total_hits;
    LOG(INFO) << "Total compared pairs " << total_compared_pairs;
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

  SharedValue<uint32_t> compared_pairs_;

  uint32_t CalculateHDEarlyBreak(
      const Stream& stream, const thrust::device_vector<point_t>& points_a,
      const thrust::device_vector<point_t>& points_b,
      ArrayView<uint32_t> v_point_ids_a = ArrayView<uint32_t>()) {
    auto* p_cmax2 = cmax2_.data();
    ArrayView<point_t> v_points_a(points_a);
    ArrayView<point_t> v_points_b(points_b);

    auto* p_compared_pairs = compared_pairs_.data();

    uint32_t n_points_a = v_point_ids_a.size();

    if (n_points_a == 0) {
      n_points_a = points_a.size();
    }

    LOG(INFO) << "Points A " << n_points_a << " Points B " << points_b.size();
    if (n_points_a < 1024 && points_b.size() > n_points_a && false) {
      LOG(INFO) << "Use Brute Force";
      compared_pairs_.set(stream.cuda_stream(),
                          points_a.size() * points_b.size());
      cmin2_.resize(n_points_a, std::numeric_limits<COORD_T>::max());
      auto* p_cmin2 = thrust::raw_pointer_cast(cmin2_.data());

      thrust::for_each(thrust::cuda::par.on(stream.cuda_stream()),
                       points_b.begin(), points_b.end(),
                       [=] __device__(const point_t& point_b) {
                         auto cmin2 = std::numeric_limits<COORD_T>::max();

                         for (size_t i = 0; i < n_points_a; i++) {
                           auto point_a_idx = i;
                           if (!v_point_ids_a.empty()) {
                             point_a_idx = v_point_ids_a[i];
                           }

                           const auto& point_a = v_points_a[point_a_idx];

                           auto d = EuclideanDistance2(point_a, point_b);
                           atomicMin(&p_cmin2[i], d);
                         }
                       });
      thrust::for_each(thrust::cuda::par.on(stream.cuda_stream()),
                       cmin2_.begin(), cmin2_.end(),
                       [=] __device__(coord_t cmin2) mutable {
                         if (cmin2 != std::numeric_limits<COORD_T>::max()) {
                           atomicMax(p_cmax2, cmin2);
                         }
                       });
    } else {
      compared_pairs_.set(stream.cuda_stream(), 0);

      LaunchKernel(stream, [=] __device__() {
        using BlockReduce = cub::BlockReduce<coord_t, MAX_BLOCK_SIZE>;
        __shared__ typename BlockReduce::TempStorage temp_storage;
        __shared__ bool early_break;
        __shared__ point_t point_a;

        for (auto i = blockIdx.x; i < n_points_a; i += gridDim.x) {
          auto size_b_roundup =
              div_round_up(v_points_b.size(), blockDim.x) * blockDim.x;
          coord_t cmin = std::numeric_limits<coord_t>::max();
          uint32_t n_pairs = 0;

          if (threadIdx.x == 0) {
            early_break = false;
            if (v_point_ids_a.empty()) {
              point_a = v_points_a[i];
            } else {
              auto point_id_s = v_point_ids_a[i];
              point_a = v_points_a[point_id_s];
            }
          }
          __syncthreads();

          for (auto j = threadIdx.x; j < size_b_roundup && !early_break;
               j += blockDim.x) {
            auto d = std::numeric_limits<coord_t>::max();
            if (j < v_points_b.size()) {
              const auto& point_b = v_points_b[j];
              d = EuclideanDistance2(point_a, point_b);
              n_pairs++;
            }

            auto agg_min = BlockReduce(temp_storage).Reduce(d, cub::Min());

            if (threadIdx.x == 0) {
              cmin = std::min(cmin, agg_min);
              if (cmin <= *p_cmax2) {
                early_break = true;
              }
            }
            __syncthreads();
          }
          atomicAdd(p_compared_pairs, n_pairs);
          __syncthreads();
          if (threadIdx.x == 0 && cmin != std::numeric_limits<coord_t>::max()) {
            atomicMax(p_cmax2, cmin);
          }
        }
      });
    }
    auto n_compared_pairs = compared_pairs_.get(stream.cuda_stream());
    // auto coefficient = n_compared_pairs / points_b.size();

    // LOG(INFO) << "EB Compared Pairs: " << n_compared_pairs
    //           << " coefficient to B " << coefficient << " cmax2 "
    //           << cmax2_.get(stream.cuda_stream());
    return n_compared_pairs;
  }

 private:
  HausdorffDistanceRTConfig config_;
  thrust::default_random_engine g_;
  thrust::device_vector<OptixAabb> aabbs_;
  thrust::device_vector<COORD_T> cmin2_;

  SharedValue<mbr_t> mbr_;
  OptixTraversableHandle gas_handle_;
  Queue<uint32_t> in_queue_, out_queue_;
  Queue<uint32_t> term_queue_;
  ReusableBuffer buffer_;
  details::RTEngine rt_engine_;
  SharedValue<COORD_T> cmax2_;
  Sampler sampler_;
};
}  // namespace hd

#endif  // HAUSDORFF_DISTANCE_RT_H
