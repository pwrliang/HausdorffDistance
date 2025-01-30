#ifndef HAUSDORFF_DISTANCE_RT_H
#define HAUSDORFF_DISTANCE_RT_H
#include <glog/logging.h>
#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <thrust/shuffle.h>

#include <iomanip>

#include "distance.h"
#include "grid.h"
#include "rt/reusable_buffer.h"
#include "rt/rt_engine.h"
#include "sampler.h"
#include "utils/bitset.h"
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

DEV_HOST_INLINE OptixAabb GetOptixAABB(float2 p, float radius, int partition,
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

    min_val = norm_min_val + partition;
    max_val = norm_max_val + partition;
  }

  return aabb;
}

DEV_HOST_INLINE OptixAabb GetOptixAABB(float3 p, float radius, int partition,
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

    min_val = norm_min_val + partition;
    max_val = norm_max_val + partition;
  }

  return aabb;
}

DEV_HOST_INLINE OptixAabb GetOptixAABB(double2 p, double radius, int partition,
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

    min_val = norm_min_val + partition;
    max_val = norm_max_val + partition;
  }

  return aabb;
}

DEV_HOST_INLINE OptixAabb GetOptixAABB(double3 p, double radius, int partition,
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

    min_val = norm_min_val + partition;
    max_val = norm_max_val + partition;
  }

  return aabb;
}

}  // namespace details

struct HausdorffDistanceRTConfig {
  const char* ptx_root;
  bool fast_build = false;
  bool compact = false;
  bool rebuild_bvh = false;
  bool shuffle = false;
  float radius_step = 2;
  int max_grid_size = 1024;
  float sample_rate = 0.05;
  int max_samples = 100 * 1000;
  bool auto_grid = true;
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
    grid_ = Grid<COORD_T, N_DIMS>(config_.max_grid_size);
    sampler_.Init(hd_config.max_samples);
  }
#if 0
  int CalculateBestGridSize(const Stream& stream,
                            const thrust::device_vector<point_t>& points_a,
                            const thrust::device_vector<point_t>& points_b,
                            const mbr_t& mbr, const mbr_t& mbr_b,
                            float& radius) {
    SharedValue<uint32_t> sv_n_near_points;
    Stopwatch sw;
    auto samples_a = sampler_.Sample(
        stream.cuda_stream(), points_a.size(),
        std::max(1u, (uint32_t) (points_a.size() * config_.sample_rate)));
    auto samples_b = sampler_.Sample(
        stream.cuda_stream(), ArrayView<point_t>(points_b),
        std::max(1u, (uint32_t) (points_b.size() * config_.sample_rate)));

    COORD_T volume = 1;
    COORD_T min_extent = std::numeric_limits<COORD_T>::max();
    COORD_T max_extent = 0;

    for (int dim = 0; dim < N_DIMS; dim++) {
      auto extent = mbr.upper(dim) - mbr.lower(dim);
      volume *= extent;
      min_extent = std::min(min_extent, extent);
      max_extent = std::max(max_extent, extent);
    }

    float density = (float) points_b.size() / volume;

    int best_grid_size = 0;
    uint64_t min_cost = std::numeric_limits<uint64_t>::max();
    double min_time = std::numeric_limits<double>::max();

    for (int grid_size = 128; grid_size <= config_.max_grid_size;) {
      sw.start();
      auto d_near_queue = near_queue_.DeviceObject();
      auto d_far_queue = far_queue_.DeviceObject();
      auto* p_points_a = thrust::raw_pointer_cast(points_a.data());

      grid_.Init(grid_size, mbr);
      grid_.Clear(stream);
      grid_.Insert(stream, samples_b);

      auto d_grid = grid_.DeviceObject();

      near_queue_.Clear(stream.cuda_stream());
      far_queue_.Clear(stream.cuda_stream());

      thrust::for_each(thrust::cuda::par.on(stream.cuda_stream()),
                       samples_a.begin(), samples_a.end(),
                       [=] __device__(uint32_t point_id) mutable {
                         const auto& p = p_points_a[point_id];
                         const auto& cell = d_grid.Query(p);

                         if (cell.n_primitives == 0) {  // far
                           d_far_queue.Append(point_id);
                         } else {  // near
                           d_near_queue.Append(point_id);
                         }
                       });
      auto near_size = near_queue_.size(stream.cuda_stream());
      auto far_size = far_queue_.size(stream.cuda_stream());

      LOG(INFO) << "Near rate " << (float) near_size / samples_a.size()
                << " Far rate " << (float) far_size / samples_a.size();

      auto eb_coefficient =
          CalculateHDEarlyBreak(
              stream, points_a, samples_b,
              ArrayView<uint32_t>(far_queue_.data(), far_size)) /
          samples_b.size();

      radius = max_extent / grid_size;
      if (near_size > 0) {
        buffer_.Clear();
        auto gas_handle = BuildBVH(stream, samples_b, radius);
        details::LaunchParamsNN<COORD_T, N_DIMS> params;

        memset(&params, 0, sizeof(params));
        params.in_queue = ArrayView<uint32_t>(near_queue_.data(), near_size);
        params.out_queue = out_queue_.DeviceObject();
        params.points_a = ArrayView<point_t>(points_a);
        params.points_b = ArrayView<point_t>(points_b);
        params.handle = gas_handle;
        params.cmax2 = cmax2_.data();
        params.radius = radius;

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
        dims.x = params.in_queue.size();
        rt_engine_.CopyLaunchParams(stream.cuda_stream(), params);
        rt_engine_.Render(stream.cuda_stream(), mod_nn, dims);
      }
      stream.Sync();
      sw.stop();
      if (sw.ms() < min_time) {
        min_time = sw.ms();
        best_grid_size = grid_size;
      }
#if 0
      auto near_rate =
          (float) near_queue_.size(stream.cuda_stream()) / samples_a.size();
      auto n_near_points = points_a.size() * near_rate;
      auto n_far_points = points_a.size() - n_near_points;
      auto s = min_extent / grid_size;
      COORD_T cube_volume = 1;

      for (int dim = 0; dim < N_DIMS; dim++) {
        cube_volume *= s;
      }

      uint64_t ray_cast_cost = n_near_points * log(points_b.size());
      uint64_t rt_intersects = n_near_points * cube_volume * density;
      uint64_t near_cost = rt_intersects;
      uint64_t far_cost = n_far_points * eb_coefficient;
      uint64_t total_cost = near_cost + far_cost;

      LOG(INFO) << "Grid " << grid_size << " s " << s << " Near Rate "
                << near_rate << " Ray Intersect Cost " << rt_intersects
                << " cube volume " << cube_volume << " density " << density
                << " RT Cost " << near_cost << " Far Cost " << far_cost
                << " Total Cost " << total_cost << " eb_coefficient "
                << eb_coefficient;

      if (total_cost < min_cost) {
        min_cost = total_cost;
        best_grid_size = grid_size;
        radius = s;
      }
#endif
      LOG(INFO) << "Grid " << grid_size << " Time " << sw.ms() << " ms";
      grid_size *= config_.radius_step;
    }

    LOG(INFO) << "best grid size : " << best_grid_size;
    return best_grid_size;
  }
#endif

  int CalculateBestGridSize(const Stream& stream,
                            const thrust::device_vector<point_t>& points_a,
                            const thrust::device_vector<point_t>& points_b,
                            const mbr_t& mbr, const mbr_t& mbr_b,
                            float& radius) {
    SharedValue<uint32_t> sv_n_near_points;
    Stopwatch sw;
    auto samples_a = sampler_.Sample(
        stream.cuda_stream(), points_a.size(),
        std::max(1u, (uint32_t) (points_a.size() * config_.sample_rate)));
    auto samples_b = sampler_.Sample(
        stream.cuda_stream(), ArrayView<point_t>(points_b),
        std::max(1u, (uint32_t) (points_b.size() * config_.sample_rate)));

    COORD_T volume = 1;
    COORD_T min_extent = std::numeric_limits<COORD_T>::max();
    COORD_T max_extent = 0;

    for (int dim = 0; dim < N_DIMS; dim++) {
      auto extent = mbr.upper(dim) - mbr.lower(dim);
      volume *= extent;
      min_extent = std::min(min_extent, extent);
      max_extent = std::max(max_extent, extent);
    }

    float density = (float) points_b.size() / volume;

    int best_grid_size = 0;
    uint64_t min_cost = std::numeric_limits<uint64_t>::max();

    for (int grid_size = 128; grid_size <= config_.max_grid_size;
         grid_size *= config_.radius_step) {
      sw.start();
      auto d_near_queue = near_queue_.DeviceObject();
      auto d_far_queue = far_queue_.DeviceObject();
      auto* p_points_a = thrust::raw_pointer_cast(points_a.data());

      grid_.Init(grid_size, mbr);
      grid_.Clear(stream);
      grid_.Insert(stream, samples_b);

      auto d_grid = grid_.DeviceObject();

      near_queue_.Clear(stream.cuda_stream());
      far_queue_.Clear(stream.cuda_stream());

      thrust::for_each(thrust::cuda::par.on(stream.cuda_stream()),
                       samples_a.begin(), samples_a.end(),
                       [=] __device__(uint32_t point_id) mutable {
                         const auto& p = p_points_a[point_id];
                         const auto& cell = d_grid.Query(p);

                         if (cell.n_primitives == 0) {  // far
                           d_far_queue.Append(point_id);
                         } else {  // near
                           d_near_queue.Append(point_id);
                         }
                       });
      auto near_size = near_queue_.size(stream.cuda_stream());
      auto far_size = far_queue_.size(stream.cuda_stream());

      LOG(INFO) << "Near rate " << (float) near_size / samples_a.size()
                << " Far rate " << (float) far_size / samples_a.size();

      cmax2_.set(stream.cuda_stream(), 0);
      auto eb_coefficient =
          CalculateHDEarlyBreak(
              stream, points_a, samples_b,
              ArrayView<uint32_t>(far_queue_.data(), far_size)) /
          samples_b.size();

      auto near_rate =
          (float) near_queue_.size(stream.cuda_stream()) / samples_a.size();
      auto n_near_points = points_a.size() * near_rate;
      auto n_far_points = points_a.size() - n_near_points;
      auto s = min_extent / grid_size;
      COORD_T cube_volume = 1;

      for (int dim = 0; dim < N_DIMS; dim++) {
        cube_volume *= s;
      }

      uint64_t ray_cast_cost = n_near_points * log(points_b.size());
      uint64_t rt_intersects = n_near_points * cube_volume * density;
      uint64_t near_cost = rt_intersects;
      uint64_t far_cost = n_far_points * eb_coefficient;
      uint64_t total_cost = near_cost + far_cost;

      LOG(INFO) << "Grid " << grid_size << " s " << s << " Near Rate "
                << near_rate << " Ray Intersect Cost " << rt_intersects
                << " cube volume " << cube_volume << " density " << density
                << " RT Cost " << near_cost << " Far Cost " << far_cost
                << " Total Cost " << total_cost << " eb_coefficient "
                << eb_coefficient;

      if (total_cost < min_cost) {
        min_cost = total_cost;
        best_grid_size = grid_size;
        radius = s;
      }
    }

    LOG(INFO) << "best grid size : " << best_grid_size;
    return best_grid_size;
  }

  COORD_T CalculateDistanceNearFar(const Stream& stream,
                                   thrust::device_vector<point_t>& points_a,
                                   thrust::device_vector<point_t>& points_b) {
    Stopwatch sw;
    auto n_points_a = points_a.size();
    auto n_points_b = points_b.size();
    const auto mbr_a = CalculateMbr(stream, points_a.begin(), points_a.end());
    const auto mbr_b = CalculateMbr(stream, points_b.begin(), points_b.end());
    auto mem_bytes = rt_engine_.EstimateMemoryUsageForAABB(
        n_points_b, config_.fast_build, config_.compact);

    buffer_.Init(mem_bytes * 1.5);
    buffer_.Clear();

    thrust::shuffle(thrust::cuda::par.on(stream.cuda_stream()),
                    points_a.begin(), points_a.end(), g_);
    thrust::shuffle(thrust::cuda::par.on(stream.cuda_stream()),
                    points_b.begin(), points_b.end(), g_);

    HdBounds<COORD_T, N_DIMS> hd_bounds(mbr_a);
    auto hd_lb = hd_bounds.GetLowerBound(mbr_b);
    auto hd_ub = hd_bounds.GetUpperBound(mbr_b);

    auto union_mbr = mbr_a;  // largest possible range
    coord_t min_extent = std::numeric_limits<coord_t>::max();
    coord_t max_extent = 0;

    union_mbr.Expand(mbr_b);

    for (int dim = 0; dim < N_DIMS; dim++) {
      auto lower = union_mbr.lower(dim) - hd_ub;
      auto upper = union_mbr.upper(dim) + hd_ub;
      union_mbr.set_lower(dim, lower);
      union_mbr.set_upper(dim, upper);
      min_extent = std::min(min_extent, upper - lower);
      max_extent = std::max(max_extent, upper - lower);
    }

    COORD_T volume = 1;
    for (int dim = 0; dim < N_DIMS; dim++) {
      auto lower = mbr_b.lower(dim);
      auto upper = mbr_b.upper(dim);
      auto extent = upper - lower;
      min_extent = std::min(min_extent, extent);
      max_extent = std::max(max_extent, extent);
      volume *= extent;
    }

    auto density = points_b.size() / volume;

    LOG(INFO) << "LB " << hd_lb << " UB " << hd_ub << " min extent "
              << min_extent << " max extent " << max_extent;

    in_queue_.Init(n_points_a);
    out_queue_.Init(n_points_a);
    near_queue_.Init(n_points_a);
    far_queue_.Init(n_points_a);
    in_queue_.Clear(stream.cuda_stream());
    out_queue_.Clear(stream.cuda_stream());

    int grid_size = config_.max_grid_size;
    float radius;

    if (config_.auto_grid) {
      grid_size = CalculateBestGridSize(stream, points_a, points_b, union_mbr,
                                        mbr_b, radius);
    } else {
      if (hd_lb > 0) {
        radius = hd_lb;
      } else {
        radius = hd_ub / 100;
      }
    }

    LOG(INFO) << "Grid " << grid_size << " radius " << radius;

    grid_.Init(grid_size, union_mbr);
    grid_.Clear(stream);
    grid_.Insert(stream, points_b);
    auto d_grid = grid_.DeviceObject();
    auto d_in_queue = in_queue_.DeviceObject();

    thrust::for_each(thrust::cuda::par.on(stream.cuda_stream()),
                     thrust::make_counting_iterator<uint32_t>(0),
                     thrust::make_counting_iterator<uint32_t>(points_a.size()),
                     [=] __device__(uint32_t point_id) mutable {
                       d_in_queue.Append(point_id);
                     });

    uint32_t in_size = n_points_a;

    buffer_.Clear();
    OptixTraversableHandle gas_handle = 0;

    SharedValue<uint32_t> skip_count;
    SharedValue<uint32_t> skip_total_idx;
    SharedValue<uint32_t> iter_hits;
    SharedValue<uint32_t> n_compared_pairs;
    int iter = 0;
    uint64_t total_hits = 0;
    uint64_t total_compared_pairs = 0;

    cmax2_.set(stream.cuda_stream(), hd_lb * hd_lb);

    while (in_size > 0) {
      iter++;
      iter_hits.set(stream.cuda_stream(), 0);
      n_compared_pairs.set(stream.cuda_stream(), 0);

      sw.start();
      near_queue_.Clear(stream.cuda_stream());
      far_queue_.Clear(stream.cuda_stream());

      auto d_near_queue = near_queue_.DeviceObject();
      auto d_far_queue = far_queue_.DeviceObject();
      auto* p_points_a = thrust::raw_pointer_cast(points_a.data());

      thrust::for_each(thrust::cuda::par.on(stream.cuda_stream()),
                       in_queue_.data(), in_queue_.data() + in_size,
                       [=] __device__(uint32_t point_id) mutable {
                         const auto& p = p_points_a[point_id];
                         const auto& cell = d_grid.Query(p);

                         if (cell.n_primitives == 0) {  // far
                           d_far_queue.Append(point_id);
                         } else {  // near
                           d_near_queue.Append(point_id);
                         }
                       });

      auto near_size = near_queue_.size(stream.cuda_stream());
      auto far_size = far_queue_.size(stream.cuda_stream());
      sw.stop();

      VLOG(1) << "Near Rate " << (float) near_size / in_size << ", Far Rate "
              << (float) far_size / in_size;

      if (far_size > 0) {
        sw.start();
        CalculateHDEarlyBreak(stream, points_a, points_b,
                              ArrayView<uint32_t>(far_queue_.data(), far_size));
        stream.Sync();
        sw.stop();
        LOG(INFO) << "EB Time " << sw.ms();
      }

      if (near_size > 0) {
        // Estimate Hits
        COORD_T primitive_volume = 1;

        for (int dim = 0; dim < N_DIMS; dim++) {
          primitive_volume *= radius;
        }

        auto estimated_hits = near_size * density * primitive_volume;

        details::LaunchParamsNN<COORD_T, N_DIMS> params;

        skip_count.set(stream.cuda_stream(), 0);
        skip_total_idx.set(stream.cuda_stream(), 0);

        if (gas_handle == 0 || !config_.rebuild_bvh) {
          buffer_.Clear();
          gas_handle = BuildBVH(stream, points_b, radius);
        } else {
          gas_handle = UpdateBVH(stream, gas_handle, points_b, radius);
        }

        params.in_queue = ArrayView<uint32_t>(near_queue_.data(), near_size);
        params.out_queue = out_queue_.DeviceObject();
        params.points_a = ArrayView<point_t>(points_a);
        params.points_b = ArrayView<point_t>(points_b);
        params.handle = gas_handle;
        params.cmax2 = cmax2_.data();
        params.radius = radius;
        params.skip_count = skip_count.data();
        params.skip_total_idx = skip_total_idx.data();
        params.n_hits = iter_hits.data();
        params.n_compared_pairs = n_compared_pairs.data();

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
        sw.start();
        dims.x = params.in_queue.size();
        rt_engine_.CopyLaunchParams(stream.cuda_stream(), params);
        rt_engine_.Render(stream.cuda_stream(), mod_nn, dims);
        stream.Sync();
        sw.stop();

        auto cmax2 = cmax2_.get(stream.cuda_stream());
        auto cmax = sqrt(cmax2);
        auto n_hits = iter_hits.get(stream.cuda_stream());
        auto n_pairs = n_compared_pairs.get(stream.cuda_stream());

        total_hits += n_hits;
        total_compared_pairs += n_pairs;

        VLOG(1) << "Iter: " << iter << " radius: " << radius << std::fixed
                << std::setprecision(8) << " cmax2: " << cmax2
                << " cmax: " << cmax << " near_size: " << near_size
                << " out_size: " << out_queue_.size(stream.cuda_stream())
                << " Iter hits " << n_hits << " Estimated hits "
                << estimated_hits << " Iter compared pairs " << n_pairs
                << " Avg hits " << (float) n_hits / near_size << " Skip idx: "
                << (float) skip_total_idx.get(stream.cuda_stream()) /
                       skip_count.get(stream.cuda_stream())
                << " Time: " << sw.ms() << " ms";

        radius *= config_.radius_step;
      }

      in_queue_.Clear(stream.cuda_stream());
      in_queue_.Swap(out_queue_);
      in_size = in_queue_.size(stream.cuda_stream());
    }
    LOG(INFO) << "Total Hits " << total_hits;
    LOG(INFO) << "Total compared pairs " << total_compared_pairs;
    return sqrt(cmax2_.get(stream.cuda_stream()));
  }

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

    if (config_.shuffle) {
      thrust::shuffle(thrust::cuda::par.on(stream.cuda_stream()),
                      points_a.begin(), points_a.end(), g_);
      thrust::shuffle(thrust::cuda::par.on(stream.cuda_stream()),
                      points_b.begin(), points_b.end(), g_);
    }

    HdBounds<COORD_T, N_DIMS> hd_bounds(mbr_a);
    auto hd_lb = hd_bounds.GetLowerBound(mbr_b);
    auto hd_ub = hd_bounds.GetUpperBound(mbr_b);
    COORD_T radius = hd_lb;

    if (radius == 0) {
      radius = hd_ub / 100;
    }

    radius *= sqrt(N_DIMS);  // Refer RTNN Fig10c
    COORD_T max_radius = hd_ub;

    LOG(INFO) << "LB " << hd_lb << " UB " << hd_ub;

    auto union_mbr = mbr_a;
    COORD_T min_extent = std::numeric_limits<COORD_T>::max();
    COORD_T max_extent = 0;
    union_mbr.Expand(mbr_b);

    for (int dim = 0; dim < N_DIMS; dim++) {
      auto lower = union_mbr.lower(dim) - max_radius;
      auto upper = union_mbr.upper(dim) + max_radius;
      union_mbr.set_lower(dim, lower);
      union_mbr.set_upper(dim, upper);
      min_extent = std::min(min_extent, upper - lower);
      max_extent = std::max(max_extent, upper - lower);
    };

    in_queue_.Init(n_points_a);
    near_queue_.Init(n_points_a);
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

    SharedValue<uint32_t> skip_count;
    SharedValue<uint32_t> skip_total_idx;
    SharedValue<uint32_t> iter_hits;
    SharedValue<uint32_t> n_compared_pairs;
    int iter = 0;
    uint64_t total_hits = 0;
    uint64_t total_compared_pairs = 0;

    cmax2_.set(stream.cuda_stream(), hd_lb * hd_lb);

    while (in_size > 0) {
      iter++;
      iter_hits.set(stream.cuda_stream(), 0);
      n_compared_pairs.set(stream.cuda_stream(), 0);
      details::LaunchParamsNN<COORD_T, N_DIMS> params;

      skip_count.set(stream.cuda_stream(), 0);
      skip_total_idx.set(stream.cuda_stream(), 0);
      ArrayView<uint32_t> in(in_queue_.data(), in_size);

      params.in_queue = in;
      params.out_queue = out_queue_.DeviceObject();
      params.points_a = ArrayView<point_t>(points_a);
      params.points_b = ArrayView<point_t>(points_b);
      params.handle = gas_handle;
      params.cmax2 = cmax2_.data();
      params.radius = radius;
      params.skip_count = skip_count.data();
      params.skip_total_idx = skip_total_idx.data();
      params.n_hits = iter_hits.data();
      params.n_compared_pairs = n_compared_pairs.data();

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
      auto n_hits = iter_hits.get(stream.cuda_stream());
      auto n_pairs = n_compared_pairs.get(stream.cuda_stream());

      total_hits += n_hits;
      total_compared_pairs += n_pairs;

      VLOG(1) << "Iter: " << iter << " radius: " << radius << std::fixed
              << std::setprecision(8) << " cmax2: " << cmax2
              << " cmax: " << cmax << " in_size: " << in_size
              << " out_size: " << out_queue_.size(stream.cuda_stream())
              << " Iter hits " << n_hits << " Iter compared pairs " << n_pairs
              << " Avg hits " << (float) n_hits / in_size << " Skip idx: "
              << (float) skip_total_idx.get(stream.cuda_stream()) /
                     skip_count.get(stream.cuda_stream())
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
    LOG(INFO) << "Total Hits " << total_hits;
    LOG(INFO) << "Total compared pairs " << total_compared_pairs;
    return sqrt(cmax2_.get(stream.cuda_stream()));
  }

  COORD_T CalculateDistance(const Stream& stream,
                            thrust::device_vector<point_t>& points_a,
                            thrust::device_vector<point_t>& points_b,
                            unsigned int parallelism) {
    auto n_points_a = points_a.size();
    auto n_points_b = points_b.size();
    auto mbr_a = CalculateMbr(stream, points_a.begin(), points_a.end());
    auto mbr_b = CalculateMbr(stream, points_b.begin(), points_b.end());
    auto mem_bytes = rt_engine_.EstimateMemoryUsageForAABB(
        n_points_b, config_.fast_build, config_.compact);

    buffer_.Init(mem_bytes * 1.5);
    buffer_.Clear();

    if (config_.shuffle) {
      thrust::shuffle(thrust::cuda::par.on(stream.cuda_stream()),
                      points_a.begin(), points_a.end(), g_);
      thrust::shuffle(thrust::cuda::par.on(stream.cuda_stream()),
                      points_b.begin(), points_b.end(), g_);
    }

    HdBounds<COORD_T, N_DIMS> hd_bounds(mbr_a);
    auto hd_lb = hd_bounds.GetLowerBound(mbr_b);
    auto hd_ub = hd_bounds.GetUpperBound(mbr_b);
    COORD_T radius = hd_lb * sqrt(N_DIMS);  // Refer RTNN Fig10c
    COORD_T max_radius = hd_ub;

    in_queue_.Init(n_points_a);
    out_queue_.Init(n_points_a);
    in_queue_.Clear(stream.cuda_stream());
    out_queue_.Clear(stream.cuda_stream());

    cmin2_.resize(n_points_a);
    thread_counters_.resize(n_points_a);

    auto d_in_queue = in_queue_.DeviceObject();

    thrust::for_each(thrust::cuda::par.on(stream.cuda_stream()),
                     thrust::make_counting_iterator<uint32_t>(0),
                     thrust::make_counting_iterator<uint32_t>(points_a.size()),
                     [=] __device__(uint32_t point_id) mutable {
                       d_in_queue.Append(point_id);
                     });

    uint32_t in_size = n_points_a;
    LOG(INFO) << "LB " << hd_lb << " UB " << hd_ub;

    buffer_.Clear();
    Mbr<coord_t, N_DIMS> union_mbr = mbr_a;
    union_mbr.Expand(mbr_b);

    for (int dim = 0; dim < N_DIMS; dim++) {
      auto lower = union_mbr.lower(dim) - max_radius;
      auto upper = union_mbr.upper(dim) + max_radius;
      union_mbr.set_lower(dim, lower);
      union_mbr.set_upper(dim, upper);
    }

    auto gas_handle =
        BuildBVH(stream, points_b, union_mbr, radius, parallelism);

    SharedValue<uint32_t> skip_count;
    SharedValue<uint32_t> skip_total_idx;
    SharedValue<uint32_t> iter_hits;
    SharedValue<uint32_t> n_compared_pairs;
    int iter = 0;
    uint64_t total_hits = 0;
    uint64_t total_compared_pairs = 0;

    cmax2_.set(stream.cuda_stream(), hd_lb * hd_lb);

    while (in_size > 0) {
      iter++;
      iter_hits.set(stream.cuda_stream(), 0);
      n_compared_pairs.set(stream.cuda_stream(), 0);

      thrust::fill_n(thrust::cuda::par.on(stream.cuda_stream()), cmin2_.begin(),
                     in_size, std::numeric_limits<COORD_T>::max());
      thrust::fill_n(thrust::cuda::par.on(stream.cuda_stream()),
                     thread_counters_.begin(), in_size, 0);

      details::LaunchParamsNNMultiCast<COORD_T, N_DIMS> params;

      skip_count.set(stream.cuda_stream(), 0);
      skip_total_idx.set(stream.cuda_stream(), 0);

      params.in_queue = ArrayView<uint32_t>(in_queue_.data(), in_size);
      params.out_queue = out_queue_.DeviceObject();
      params.points_a = ArrayView<point_t>(points_a);
      params.points_b = ArrayView<point_t>(points_b);
      params.handle = gas_handle;
      params.cmin2 = thrust::raw_pointer_cast(cmin2_.data());
      params.cmax2 = cmax2_.data();
      params.radius = radius;
      params.skip_count = skip_count.data();
      params.skip_total_idx = skip_total_idx.data();
      params.n_hits = iter_hits.data();
      params.n_compared_pairs = n_compared_pairs.data();
      params.mbr = union_mbr;
      params.thread_counters =
          thrust::raw_pointer_cast(thread_counters_.data());

      details::ModuleIdentifier mod_nn = details::NUM_MODULE_IDENTIFIERS;

      if (typeid(COORD_T) == typeid(float)) {
        if (N_DIMS == 2) {
          mod_nn = details::MODULE_ID_FLOAT_NN_MULTICAST_2D;
        } else if (N_DIMS == 3) {
          mod_nn = details::MODULE_ID_FLOAT_NN_MULTICAST_3D;
        }
      } else if (typeid(COORD_T) == typeid(double)) {
        if (N_DIMS == 2) {
          mod_nn = details::MODULE_ID_DOUBLE_NN_MULTICAST_2D;
        } else if (N_DIMS == 3) {
          mod_nn = details::MODULE_ID_DOUBLE_NN_MULTICAST_3D;
        }
      }

      dim3 dims{1, 1, 1};
      Stopwatch sw;
      sw.start();
      dims.x = in_size;
      dims.y = parallelism;
      uint32_t max_size = 1 << 30;

      if (dims.x * dims.y > max_size) {
        dims.x = max_size / dims.y;
      }
      rt_engine_.CopyLaunchParams(stream.cuda_stream(), params);
      rt_engine_.Render(stream.cuda_stream(), mod_nn, dims);
      stream.Sync();
      sw.stop();

      auto cmax2 = cmax2_.get(stream.cuda_stream());
      auto cmax = sqrt(cmax2);
      auto n_hits = iter_hits.get(stream.cuda_stream());
      auto n_pairs = n_compared_pairs.get(stream.cuda_stream());
      total_hits += n_hits;
      total_compared_pairs += n_pairs;

      VLOG(1) << "Iter: " << iter << " radius: " << radius << std::fixed
              << std::setprecision(8) << " cmax2: " << cmax2
              << " cmax: " << cmax << " in_size: " << in_size
              << " out_size: " << out_queue_.size(stream.cuda_stream())
              << " Iter hits " << n_hits << " Iter compared pairs " << n_pairs
              << " Avg hits " << (float) n_hits / in_size << " Skip idx: "
              << (float) skip_total_idx.get(stream.cuda_stream()) /
                     skip_count.get(stream.cuda_stream())
              << " Time: " << sw.ms() << " ms";

      in_queue_.Clear(stream.cuda_stream());
      in_queue_.Swap(out_queue_);

      radius *= config_.radius_step;
      in_size = in_queue_.size(stream.cuda_stream());

      if (in_size > 0) {
        if (config_.rebuild_bvh) {
          buffer_.Clear();
          gas_handle =
              BuildBVH(stream, points_b, union_mbr, radius, parallelism);
        } else {
          gas_handle = UpdateBVH(stream, gas_handle, points_b, union_mbr,
                                 radius, parallelism);
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

  OptixTraversableHandle BuildBVH(const Stream& stream,
                                  ArrayView<point_t> points, const mbr_t& mbr,
                                  COORD_T radius, uint32_t n_partitions) {
    aabbs_.resize(points.size());
    ArrayView<OptixAabb> aabbs(aabbs_);

    thrust::for_each(thrust::cuda::par.on(stream.cuda_stream()),
                     thrust::make_counting_iterator<size_t>(0),
                     thrust::make_counting_iterator<size_t>(points.size()),
                     [=] __device__(size_t i) mutable {
                       const auto& p = points[i];
                       auto partition = i % n_partitions;
                       aabbs[i] =
                           details::GetOptixAABB(p, radius, partition, mbr);
                     });

    return rt_engine_.BuildAccelCustom(stream.cuda_stream(), aabbs, buffer_,
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

  OptixTraversableHandle UpdateBVH(const Stream& stream,
                                   OptixTraversableHandle handle,
                                   ArrayView<point_t> points, const mbr_t& mbr,
                                   COORD_T radius, uint32_t n_partitions) {
    aabbs_.resize(points.size());
    ArrayView<OptixAabb> aabbs(aabbs_);

    thrust::for_each(thrust::cuda::par.on(stream.cuda_stream()),
                     thrust::make_counting_iterator<size_t>(0),
                     thrust::make_counting_iterator<size_t>(points.size()),
                     [=] __device__(size_t i) mutable {
                       const auto& p = points[i];
                       auto partition = i % n_partitions;
                       aabbs[i] =
                           details::GetOptixAABB(p, radius, partition, mbr);
                     });
    return rt_engine_.UpdateAccelCustom(stream.cuda_stream(), handle,
                                        ArrayView<OptixAabb>(aabbs_), buffer_,
                                        0, config_.fast_build, config_.compact);
  }

  uint32_t CalculateHDEarlyBreak(
      const Stream& stream, const thrust::device_vector<point_t>& points_a,
      const thrust::device_vector<point_t>& points_b,
      ArrayView<uint32_t> v_point_ids_a = ArrayView<uint32_t>()) {
    auto* p_cmax2 = cmax2_.data();
    ArrayView<point_t> v_points_a(points_a);
    ArrayView<point_t> v_points_b(points_b);
    SharedValue<uint32_t> compared_pairs;
    auto* p_compared_pairs = compared_pairs.data();

    compared_pairs.set(stream.cuda_stream(), 0);

    uint32_t n_points_a = v_point_ids_a.size();

    if (n_points_a == 0) {
      n_points_a = points_a.size();
    }

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

    auto n_compared_pairs = compared_pairs.get(stream.cuda_stream());
    auto coefficient = n_compared_pairs / points_b.size();

    LOG(INFO) << "EB Compared Pairs: " << n_compared_pairs
              << " coefficient to B " << coefficient << " cmax2 "
              << cmax2_.get(stream.cuda_stream());
    return n_compared_pairs;
  }

 private:
  HausdorffDistanceRTConfig config_;
  thrust::default_random_engine g_;
  thrust::device_vector<OptixAabb> aabbs_;
  thrust::device_vector<COORD_T> cmin2_;
  thrust::device_vector<uint32_t> thread_counters_;

  SharedValue<mbr_t> mbr_;
  OptixTraversableHandle gas_handle_;
  Queue<uint32_t> in_queue_, out_queue_;
  ReusableBuffer buffer_;
  details::RTEngine rt_engine_;
  SharedValue<COORD_T> cmax2_;
  Grid<COORD_T, N_DIMS> grid_;
  Sampler sampler_;
  // near-far
  Queue<uint32_t> near_queue_, far_queue_;
};
}  // namespace hd

#endif  // HAUSDORFF_DISTANCE_RT_H
