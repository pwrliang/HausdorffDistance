#ifndef HAUSDORFF_DISTANCE_RT_H
#define HAUSDORFF_DISTANCE_RT_H
#include <glog/logging.h>
#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <thrust/shuffle.h>

#include <cmath>
#include <iomanip>
#include <nlohmann/json.hpp>

#include "distance.h"
#include "grid.h"
#include "hd_bounds.h"
#include "hdr/hdr_histogram.h"
#include "models/features.h"
#include "models/tree_maxhitinit_3d.h"
#include "models/tree_maxhitnext_3d.h"
#include "rt/reusable_buffer.h"
#include "rt/rt_engine.h"
#include "sampler.h"
#include "utils/derived_atomic_functions.h"
#include "utils/helpers.h"
#include "utils/markers.h"
#include "utils/queue.h"
#include "utils/stopwatch.h"
#include "utils/stream.h"
#include "utils/type_traits.h"
#include "utils/util.h"

#define NEXT_AFTER_ROUNDS (2)
// #define PROFILING

namespace hd {

namespace details {

DEV_HOST_INLINE std::uint32_t expand_bits(std::uint32_t v) noexcept {
  v = (v * 0x00010001u) & 0xFF0000FFu;
  v = (v * 0x00000101u) & 0x0F00F00Fu;
  v = (v * 0x00000011u) & 0xC30C30C3u;
  v = (v * 0x00000005u) & 0x49249249u;
  return v;
}

// Calculates a 30-bit Morton code for the
// given 3D point located within the unit cube [0,1].
DEV_HOST_INLINE std::uint32_t morton_code(float2 xy,
                                          float resolution = 1024.0f) noexcept {
  xy.x = ::fminf(::fmaxf(xy.x * resolution, 0.0f), resolution - 1.0f);
  xy.y = ::fminf(::fmaxf(xy.y * resolution, 0.0f), resolution - 1.0f);
  const std::uint32_t xx = expand_bits(static_cast<std::uint32_t>(xy.x));
  const std::uint32_t yy = expand_bits(static_cast<std::uint32_t>(xy.y));
  return xx * 2 + yy;
}

DEV_HOST_INLINE std::uint32_t morton_code(float3 xyz,
                                          float resolution = 1024.0f) noexcept {
  xyz.x = ::fminf(::fmaxf(xyz.x * resolution, 0.0f), resolution - 1.0f);
  xyz.y = ::fminf(::fmaxf(xyz.y * resolution, 0.0f), resolution - 1.0f);
  xyz.z = ::fminf(::fmaxf(xyz.z * resolution, 0.0f), resolution - 1.0f);
  const std::uint32_t xx = expand_bits(static_cast<std::uint32_t>(xyz.x));
  const std::uint32_t yy = expand_bits(static_cast<std::uint32_t>(xyz.y));
  const std::uint32_t zz = expand_bits(static_cast<std::uint32_t>(xyz.z));
  return xx * 4 + yy * 2 + zz;
}

DEV_HOST_INLINE std::uint32_t morton_code(double2 xy,
                                          double resolution = 1024.0) noexcept {
  xy.x = ::fmin(::fmax(xy.x * resolution, 0.0), resolution - 1.0);
  xy.y = ::fmin(::fmax(xy.y * resolution, 0.0), resolution - 1.0);
  const std::uint32_t xx = expand_bits(static_cast<std::uint32_t>(xy.x));
  const std::uint32_t yy = expand_bits(static_cast<std::uint32_t>(xy.y));
  return xx * 2 + yy;
}

DEV_HOST_INLINE std::uint32_t morton_code(double3 xyz,
                                          double resolution = 1024.0) noexcept {
  xyz.x = ::fmin(::fmax(xyz.x * resolution, 0.0), resolution - 1.0);
  xyz.y = ::fmin(::fmax(xyz.y * resolution, 0.0), resolution - 1.0);
  xyz.z = ::fmin(::fmax(xyz.z * resolution, 0.0), resolution - 1.0);
  const std::uint32_t xx = expand_bits(static_cast<std::uint32_t>(xyz.x));
  const std::uint32_t yy = expand_bits(static_cast<std::uint32_t>(xyz.y));
  const std::uint32_t zz = expand_bits(static_cast<std::uint32_t>(xyz.z));
  return xx * 4 + yy * 2 + zz;
}

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

// FIXME: May have precision issue for double type
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
  float sample_rate = 0.001;
  int n_points_cell = 8;
  int max_samples = 100 * 1000;
  uint32_t max_hit = 1000;
  int max_reg_count = 0;
  bool sort_rays = false;
  bool auto_tune = false;
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
    rt_config.max_reg_count = hd_config.max_reg_count;
    rt_engine_.Init(rt_config);
    sampler_ = Sampler(config_.seed);
    sampler_.Init(hd_config.max_samples);
    g_ = thrust::default_random_engine(config_.seed);
  }

  void UpdateConfig(const HausdorffDistanceRTConfig& hd_config) {
    config_ = hd_config;
  }

  const nlohmann::json& GetStats() const { return stats_; }

  COORD_T CalculateDistance(const Stream& stream,
                            thrust::device_vector<point_t>& points_a,
                            thrust::device_vector<point_t>& points_b) {
    auto n_points_a = points_a.size();
    auto n_points_b = points_b.size();
    const auto mbr_a = CalculateMbr(stream, points_a.begin(), points_a.end());
    const auto mbr_b = CalculateMbr(stream, points_b.begin(), points_b.end());

    HdBounds<COORD_T, N_DIMS> hd_bounds(mbr_a);
    auto hd_lb = hd_bounds.GetLowerBound(mbr_b);
    auto hd_ub = hd_bounds.GetUpperBound(mbr_b);

    stats_["HDLowerBound"] = hd_lb;
    stats_["HDUpperBound"] = hd_ub;
    cmax2_.set(stream.cuda_stream(), hd_lb * hd_lb);
    COORD_T radius = hd_lb;

    Stopwatch sw;

    // Sample points for a better initial HD
    {
      sw.start();
      uint32_t n_samples = ceil(points_a.size() * config_.sample_rate);
      auto sampled_point_ids_a =
          sampler_.Sample(stream.cuda_stream(), points_a.size(), n_samples);
      thrust::device_vector<point_t> backup_points_b = points_b;
      thrust::shuffle(thrust::cuda::par.on(stream.cuda_stream()),
                      points_b.begin(), points_b.end(), g_);
      CalculateHDEarlyBreak(stream, points_a, points_b, sampled_point_ids_a);
      auto sampled_hd2 = cmax2_.get(stream.cuda_stream());
      radius = sqrt(sampled_hd2);
      points_b = backup_points_b;
      sw.stop();
      stats_["NumSamples"] = n_samples;
      stats_["SampleTime"] = sw.ms();
      stats_["HD2AfterSampling"] = sampled_hd2;
    }
    if (radius == 0) {
      radius = hd_lb;
    }
    if (radius == 0) {
      radius = hd_ub / 100;
    }
    CHECK_GT(radius, 0);

    COORD_T max_radius = hd_ub;

    stats_["HDLowerBound"] = hd_lb;
    stats_["HDUpperBound"] = hd_ub;
    stats_["InitRadius"] = radius;

    in_queue_.Init(n_points_a);
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

    sw.start();
    auto mem_bytes = rt_engine_.EstimateMemoryUsageForAABB(
        n_points_b, config_.fast_build, config_.compact);
    buffer_.Init(mem_bytes * 1.2);
    buffer_.Clear();
    auto gas_handle = BuildBVH(stream, points_b, radius);
    stream.Sync();
    sw.stop();

    stats_["BVHBuildTime"] = sw.ms();
    stats_["BVHMemoryKB"] = mem_bytes / 1024;

    SharedValue<uint32_t> iter_hits;
    int iter = 0;
#ifdef PROFILING
    struct hdr_histogram* histogram;
    // Initialise the histogram
    hdr_init(1,                     // Minimum value
             (int64_t) n_points_b,  // Maximum value
             3,                     // Number of significant figures
             &histogram);           // Pointer to initialise
#endif

    while (in_size > 0) {
      iter++;
      auto& json_iter = stats_["Iter" + std::to_string(iter)];

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
#ifdef PROFILING
      hits_counters_.resize(in_size, 0);
      params.hits_counters = thrust::raw_pointer_cast(hits_counters_.data());
#else
      params.hits_counters = nullptr;
#endif
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

      dim3 dims{in_size, 1, 1};
      sw.start();
      rt_engine_.CopyLaunchParams(stream.cuda_stream(), params);
      rt_engine_.Render(stream.cuda_stream(), mod_nn, dims);
      auto cmax2 = cmax2_.get(stream.cuda_stream());
      sw.stop();

#ifdef PROFILING
      thrust::host_vector<uint32_t> h_hits_counters = hits_counters_;

      for (auto val : h_hits_counters) {
        if (val > 0)
          hdr_record_value(histogram,  // Histogram to record to
                           val);       // Value to record
      }
      json_iter["HitsHistogram"] = DumpHistogram(histogram);
      hdr_reset(histogram);
#endif

      auto cmax = sqrt(cmax2);

      VLOG(1) << "Iter: " << iter << " radius: " << radius << std::fixed
              << std::setprecision(8) << " cmax2: " << cmax2
              << " cmax: " << cmax << " in_size: " << in_size
              << " out_size: " << out_queue_.size(stream.cuda_stream())
              << " Time: " << sw.ms() << " ms";

      in_queue_.Clear(stream.cuda_stream());
      in_queue_.Swap(out_queue_);

      if (radius < max_radius) {
        radius *= config_.radius_step;
        radius = std::min(radius, max_radius);
      }
      json_iter["NumInputPoints"] = in_size;
      in_size = in_queue_.size(stream.cuda_stream());
      json_iter["NumOutputPoints"] = in_size;
      json_iter["CMax2"] = cmax2;
      json_iter["RTTime"] = sw.ms();
      json_iter["Hits"] = iter_hits.get(stream.cuda_stream());

      if (in_size > 0) {
        sw.start();
        if (config_.rebuild_bvh) {
          buffer_.Clear();
          gas_handle = BuildBVH(stream, points_b, radius);
        } else {
          gas_handle = UpdateBVH(stream, gas_handle, points_b, radius);
        }
        stream.Sync();
        sw.stop();
        json_iter["AdjustBVHTime"] = sw.ms();
      }
    }
    return sqrt(cmax2_.get(stream.cuda_stream()));
  }

  COORD_T CalculateDistanceCompress(const Stream& stream,
                                    thrust::device_vector<point_t>& points_a,
                                    thrust::device_vector<point_t>& points_b) {
    auto n_points_a = points_a.size();
    auto n_points_b = points_b.size();
    const auto mbr_a = CalculateMbr(stream, points_a.begin(), points_a.end());
    const auto mbr_b = CalculateMbr(stream, points_b.begin(), points_b.end());

    HdBounds<COORD_T, N_DIMS> hd_bounds(mbr_a);
    auto hd_lb = hd_bounds.GetLowerBound(mbr_b);
    auto hd_ub = hd_bounds.GetUpperBound(mbr_b);

    cmax2_.set(stream.cuda_stream(), hd_lb * hd_lb);
    COORD_T radius = hd_lb;

    Stopwatch sw;

    // Sample points for a better initial HD
    {
      sw.start();
      uint32_t n_samples = ceil(points_a.size() * config_.sample_rate);
      auto sampled_point_ids_a =
          sampler_.Sample(stream.cuda_stream(), points_a.size(), n_samples);
      thrust::device_vector<point_t> backup_points_b = points_b;
      thrust::shuffle(thrust::cuda::par.on(stream.cuda_stream()),
                      points_b.begin(), points_b.end(), g_);
      CalculateHDEarlyBreak(stream, points_a, points_b, sampled_point_ids_a);
      auto sampled_hd2 = cmax2_.get(stream.cuda_stream());
      radius = sqrt(sampled_hd2);
      points_b = backup_points_b;
      sw.stop();
      stats_["NumSamples"] = n_samples;
      stats_["SampleTime"] = sw.ms();
      stats_["HD2AfterSampling"] = sampled_hd2;
    }
    if (radius == 0) {
      radius = hd_lb;
    }
    if (radius == 0) {
      radius = hd_ub / 100;
    }
    CHECK_GT(radius, 0);

    COORD_T max_radius = hd_ub;

    stats_["HDLowerBound"] = hd_lb;
    stats_["HDUpperBound"] = hd_ub;
    stats_["InitRadius"] = radius;

    in_queue_.Init(n_points_a);
    out_queue_.Init(n_points_a);
    in_queue_.Clear(stream.cuda_stream());
    out_queue_.Clear(stream.cuda_stream());

    sw.start();
    auto grid_size =
        grid_.CalculateGridResolution(mbr_b, n_points_b, config_.n_points_cell);

    grid_.Init(grid_size, mbr_b);
    grid_.Insert(stream, points_b);
    auto mbrs_b = grid_.GetCellMbrs(stream);
    stream.Sync();
    sw.stop();

    stats_["Grid"] = grid_.GetStats();

    auto d_in_queue = in_queue_.DeviceObject();

    thrust::for_each(thrust::cuda::par.on(stream.cuda_stream()),
                     thrust::make_counting_iterator<uint32_t>(0),
                     thrust::make_counting_iterator<uint32_t>(points_a.size()),
                     [=] __device__(uint32_t point_id) mutable {
                       d_in_queue.Append(point_id);
                     });

    uint32_t in_size = n_points_a;
    sw.start();
    auto mem_bytes = rt_engine_.EstimateMemoryUsageForAABB(
        mbrs_b.size(), config_.fast_build, config_.compact);
    buffer_.Init(mem_bytes * 1.2);
    buffer_.Clear();
    auto gas_handle = BuildBVH(stream, mbrs_b, radius);
    stream.Sync();
    sw.stop();

    stats_["BVHBuildTime"] = sw.ms();
    stats_["BVHMemoryKB"] = mem_bytes / 1024;

    SharedValue<uint32_t> iter_hits;
    int iter = 0;
#ifdef PROFILING
    hdr_histogram* histogram;
    // Initialise the histogram
    hdr_init(1,                     // Minimum value
             (int64_t) n_points_b,  // Maximum value
             3,                     // Number of significant figures
             &histogram);           // Pointer to initialise
#endif

    while (in_size > 0) {
      iter++;
      auto& json_iter = stats_["Iter" + std::to_string(iter)];

      iter_hits.set(stream.cuda_stream(), 0);
      details::LaunchParamsNNCompress<COORD_T, N_DIMS> params;
      auto* p_points_a = thrust::raw_pointer_cast(points_a.data());
      thrust::device_vector<uint32_t> morton_codes;

      if (config_.sort_rays) {
        sw.start();
        morton_codes.resize(in_size);
        thrust::transform(thrust::cuda::par.on(stream.cuda_stream()),
                          in_queue_.data(), in_queue_.data() + in_size,
                          morton_codes.begin(), [=] __device__(uint32_t pid) {
                            const auto& p = p_points_a[pid];
                            return details::morton_code(p);
                          });
        thrust::sort_by_key(thrust::cuda::par.on(stream.cuda_stream()),
                            morton_codes.begin(), morton_codes.end(),
                            in_queue_.data());
        stream.Sync();
        sw.stop();
        json_iter["SortRaysTime"] = sw.ms();
      }

      params.in_queue = ArrayView<uint32_t>(in_queue_.data(), in_size);
      params.miss_queue = out_queue_.DeviceObject();
      params.points_a = points_a;
      params.points_b = points_b;
      params.handle = gas_handle;
      params.cmax2 = cmax2_.data();
      params.radius = radius;
      params.mbrs_b = mbrs_b;
      params.prefix_sum = grid_.get_prefix_sum();
      params.point_b_ids = grid_.get_point_ids();
      params.n_hits = iter_hits.data();
#ifdef PROFILING
      hits_counters_.resize(in_size, 0);
      params.hits_counters = thrust::raw_pointer_cast(hits_counters_.data());
#else
      params.hits_counters = nullptr;
#endif
      params.max_hit = std::numeric_limits<uint32_t>::max();

      details::ModuleIdentifier mod_nn = details::NUM_MODULE_IDENTIFIERS;

      if (typeid(COORD_T) == typeid(float)) {
        if (N_DIMS == 2) {
          mod_nn = details::MODULE_ID_FLOAT_NN_COMPRESS_2D;
        } else if (N_DIMS == 3) {
          mod_nn = details::MODULE_ID_FLOAT_NN_COMPRESS_3D;
        }
      } else if (typeid(COORD_T) == typeid(double)) {
        if (N_DIMS == 2) {
          mod_nn = details::MODULE_ID_DOUBLE_NN_COMPRESS_2D;
        } else if (N_DIMS == 3) {
          mod_nn = details::MODULE_ID_DOUBLE_NN_COMPRESS_3D;
        }
      }

      dim3 dims{in_size, 1, 1};
      sw.start();
      rt_engine_.CopyLaunchParams(stream.cuda_stream(), params);
      rt_engine_.Render(stream.cuda_stream(), mod_nn, dims);
      stream.Sync();
      sw.stop();

#ifdef PROFILING
      thrust::host_vector<uint32_t> h_hits_counters = hits_counters_;

      for (auto val : h_hits_counters) {
        if (val > 0)
          hdr_record_value(histogram,  // Histogram to record to
                           val);       // Value to record
      }
      json_iter["HitsHistogram"] = DumpHistogram(histogram);
      hdr_reset(histogram);
#endif

      in_queue_.Clear(stream.cuda_stream());
      in_queue_.Swap(out_queue_);

      json_iter["NumInputPoints"] = in_size;
      in_size = in_queue_.size(stream.cuda_stream());
      json_iter["NumOutputPoints"] = in_size;
      json_iter["CMax2"] = cmax2_.get(stream.cuda_stream());
      json_iter["RTTime"] = sw.ms();
      json_iter["Hits"] = iter_hits.get(stream.cuda_stream());

      if (radius < max_radius) {
        radius *= config_.radius_step;
        radius = std::min(radius, max_radius);
      }
      if (in_size > 0) {
        sw.start();
        if (config_.rebuild_bvh) {
          buffer_.Clear();
          gas_handle = BuildBVH(stream, mbrs_b, radius);
        } else {
          gas_handle = UpdateBVH(stream, gas_handle, mbrs_b, radius);
        }
        stream.Sync();
        sw.stop();
        json_iter["AdjustBVHTime"] = sw.ms();
      }
    }
#ifdef PROFILING
    hdr_close(histogram);
#endif

    return sqrt(cmax2_.get(stream.cuda_stream()));
  }

  COORD_T CalculateDistanceHybrid(const Stream& stream,
                                  thrust::device_vector<point_t>& points_a,
                                  thrust::device_vector<point_t>& points_b) {
    auto n_points_a = points_a.size();
    auto n_points_b = points_b.size();
    const auto mbr_a = CalculateMbr(stream, points_a.begin(), points_a.end());
    const auto mbr_b = CalculateMbr(stream, points_b.begin(), points_b.end());

    // N.B., Do not shuffle A for better ray-coherence

    HdBounds<COORD_T, N_DIMS> hd_bounds(mbr_a);
    auto hd_lb = hd_bounds.GetLowerBound(mbr_b);
    auto hd_ub = hd_bounds.GetUpperBound(mbr_b);
    COORD_T radius;
    thrust::device_vector<point_t> points_b_shuffled = points_b;

    cmax2_.set(stream.cuda_stream(), hd_lb * hd_lb);
    thrust::shuffle(thrust::cuda::par.on(stream.cuda_stream()),
                    points_b_shuffled.begin(), points_b_shuffled.end(), g_);

    Stopwatch sw;
    stats_.clear();
    // Sample points for a better initial HD
    // Compute with grid + LB
    {
      sw.start();
      uint32_t n_samples = ceil(points_a.size() * config_.sample_rate);
      auto sampled_point_ids_a =
          sampler_.Sample(stream.cuda_stream(), points_a.size(), n_samples);
      CalculateHDEarlyBreak(stream, points_a, points_b_shuffled,
                            sampled_point_ids_a);
      auto sampled_hd2 = cmax2_.get(stream.cuda_stream());
      radius = sqrt(sampled_hd2);
      sw.stop();
      stats_["NumSamples"] = n_samples;
      stats_["SampleTime"] = sw.ms();
      stats_["HD2AfterSampling"] = sampled_hd2;
    }
    if (radius == 0) {
      radius = hd_lb;
    }
    if (radius == 0) {
      radius = hd_ub / 100;
    }
    CHECK_GT(radius, 0);

    COORD_T max_radius = hd_ub;

    stats_["HDLowerBound"] = hd_lb;
    stats_["HDUpperBound"] = hd_ub;
    stats_["InitRadius"] = radius;

    in_queue_.Init(n_points_a);
    term_queue_.Init(n_points_a);
    out_queue_.Init(n_points_a);
    in_queue_.Clear(stream.cuda_stream());
    out_queue_.Clear(stream.cuda_stream());

    sw.start();
    auto grid_size =
        grid_.CalculateGridResolution(mbr_b, n_points_b, config_.n_points_cell);
    grid_.Init(grid_size, mbr_b);
    grid_.Insert(stream, points_b);
    auto mbrs_b = grid_.GetCellMbrs(stream);
    stream.Sync();
    sw.stop();

    stats_["Grid"] = grid_.GetStats();

    auto d_in_queue = in_queue_.DeviceObject();

    thrust::for_each(thrust::cuda::par.on(stream.cuda_stream()),
                     thrust::make_counting_iterator<uint32_t>(0),
                     thrust::make_counting_iterator<uint32_t>(points_a.size()),
                     [=] __device__(uint32_t point_id) mutable {
                       d_in_queue.Append(point_id);
                     });

    uint32_t in_size = n_points_a;
    sw.start();
    auto mem_bytes = rt_engine_.EstimateMemoryUsageForAABB(
        mbrs_b.size(), config_.fast_build, config_.compact);
    buffer_.Init(mem_bytes * 1.2);
    buffer_.Clear();
    auto gas_handle = BuildBVH(stream, mbrs_b, radius);
    stream.Sync();
    sw.stop();

    stats_["BVHBuildTime"] = sw.ms();
    stats_["BVHMemoryKB"] = mem_bytes / 1024;

    std::string result = stats_.dump(4);

    thrust::device_vector<uint32_t> morton_codes;

    SharedValue<uint32_t> iter_hits;
    int iter = 0;
#ifdef PROFILING
    struct hdr_histogram* histogram;
    // Initialise the histogram
    hdr_init(1,                     // Minimum value
             (int64_t) n_points_b,  // Maximum value
             3,                     // Number of significant figures
             &histogram);           // Pointer to initialise
#endif

    // Predict MaxHit
    auto& global_stats = RunningStats::instance().Get("Input");
    FeaturesMaxHitInit<N_DIMS> features_max_hit_init(global_stats);
    FeaturesMaxHitNext<N_DIMS> features_max_hit_next(global_stats);

    features_max_hit_init.UpdateRunningInfo(stats_);

    double predict_max_hit_init;

    if (N_DIMS == 3) {
      auto features = features_max_hit_init.Serialize();
      predict_max_hit_init = PredictMaxHitInit_3D(features.data());
      if (config_.auto_tune) {
        config_.max_hit = predict_max_hit_init;
      }
    }

    while (in_size > 0) {
      iter++;
      auto& json_iter = stats_["Iter" + std::to_string(iter)];

      iter_hits.set(stream.cuda_stream(), 0);
      term_queue_.Clear(stream.cuda_stream());

      details::LaunchParamsNNCompress<COORD_T, N_DIMS> params;
      auto* p_points_a = thrust::raw_pointer_cast(points_a.data());

      if (config_.sort_rays) {
        sw.start();
        morton_codes.resize(in_size);
        thrust::transform(thrust::cuda::par.on(stream.cuda_stream()),
                          in_queue_.data(), in_queue_.data() + in_size,
                          morton_codes.begin(), [=] __device__(uint32_t pid) {
                            const auto& p = p_points_a[pid];
                            return details::morton_code(p);
                          });
        thrust::sort_by_key(thrust::cuda::par.on(stream.cuda_stream()),
                            morton_codes.begin(), morton_codes.end(),
                            in_queue_.data());
        stream.Sync();
        sw.stop();
        json_iter["SortRaysTime"] = sw.ms();
      }

      params.in_queue = ArrayView<uint32_t>(in_queue_.data(), in_size);
      params.term_queue = term_queue_.DeviceObject();
      params.miss_queue = out_queue_.DeviceObject();
      params.points_a = points_a;
      params.points_b = points_b;
      params.handle = gas_handle;
      params.cmax2 = cmax2_.data();
      params.radius = radius;
      params.mbrs_b = mbrs_b;
      params.prefix_sum = grid_.get_prefix_sum();
      params.point_b_ids = grid_.get_point_ids();
      params.n_hits = iter_hits.data();
#ifdef PROFILING
      hits_counters_.resize(in_size, 0);
      params.hits_counters = thrust::raw_pointer_cast(hits_counters_.data());
#else
      params.hits_counters = nullptr;
#endif
      params.max_hit = config_.max_hit;

      details::ModuleIdentifier mod_nn = details::NUM_MODULE_IDENTIFIERS;

      if (typeid(COORD_T) == typeid(float)) {
        if (N_DIMS == 2) {
          mod_nn = details::MODULE_ID_FLOAT_NN_COMPRESS_2D;
        } else if (N_DIMS == 3) {
          mod_nn = details::MODULE_ID_FLOAT_NN_COMPRESS_3D;
        }
      } else if (typeid(COORD_T) == typeid(double)) {
        if (N_DIMS == 2) {
          mod_nn = details::MODULE_ID_DOUBLE_NN_COMPRESS_2D;
        } else if (N_DIMS == 3) {
          mod_nn = details::MODULE_ID_DOUBLE_NN_COMPRESS_3D;
        }
      }

      ArrayView<uint32_t> eb_point_a_ids;

      dim3 dims{in_size, 1, 1};
      sw.start();
      rt_engine_.CopyLaunchParams(stream.cuda_stream(), params);
      rt_engine_.Render(stream.cuda_stream(), mod_nn, dims);
      auto miss_size = out_queue_.size(stream.cuda_stream());
      auto term_size = term_queue_.size(stream.cuda_stream());
      eb_point_a_ids = ArrayView<uint32_t>(term_queue_.data(), term_size);
      sw.stop();

      VLOG(1) << "RT Time " << sw.ms() << " In " << in_size << " miss " << miss_size << " terms " << term_size;

#ifdef PROFILING
      thrust::host_vector<uint32_t> h_hits_counters = hits_counters_;

      for (auto val : h_hits_counters) {
        if (val > 0)
          hdr_record_value(histogram,  // Histogram to record to
                           val);       // Value to record
      }
      json_iter["HitsHistogram"] = DumpHistogram(histogram);
      hdr_reset(histogram);
#endif

      json_iter["MaxHit"] = config_.max_hit;
      json_iter["NumInputPoints"] = in_size;
      json_iter["NumOutputPoints"] = miss_size;
      json_iter["NumTermPoints"] = term_size;
      json_iter["CMax2"] = cmax2_.get(stream.cuda_stream());
      json_iter["RTTime"] = sw.ms();
      json_iter["Hits"] = iter_hits.get(stream.cuda_stream());

      if (!eb_point_a_ids.empty()) {
        sw.start();
        thrust::shuffle(thrust::cuda::par.on(stream.cuda_stream()),
                        eb_point_a_ids.begin(), eb_point_a_ids.end(), g_);
        auto eb_compared_pairs = CalculateHDEarlyBreak(
            stream, points_a, points_b_shuffled, eb_point_a_ids);
        sw.stop();
        json_iter["ComparedPairs"] = eb_compared_pairs;
        json_iter["EBTime"] = sw.ms();
        VLOG(1) << "EB Time " << sw.ms();
      } else {
        json_iter["ComparedPairs"] = 0;
        json_iter["EBTime"] = 0;
      }



      in_queue_.Clear(stream.cuda_stream());
      in_queue_.Swap(out_queue_);

      if (radius < max_radius) {
        radius *= config_.radius_step;
        radius = std::min(radius, max_radius);
      }
      in_size = in_queue_.size(stream.cuda_stream());
      if (in_size > 0) {
        sw.start();
        if (config_.rebuild_bvh) {
          buffer_.Clear();
          gas_handle = BuildBVH(stream, mbrs_b, radius);
        } else {
          gas_handle = UpdateBVH(stream, gas_handle, mbrs_b, radius);
        }
        stream.Sync();
        sw.stop();
        json_iter["AdjustBVHTime"] = sw.ms();
      }

      if (config_.auto_tune) {
        features_max_hit_next.UpdateRunningInfo(json_iter);
        auto features = features_max_hit_next.Serialize();
        auto max_hit = PredictMaxHitNext_3D(features.data());
        config_.max_hit = max_hit;
      }
    }
    return sqrt(cmax2_.get(stream.cuda_stream()));
  }

  COORD_T CalculateDistanceHybrid(const Stream& stream,
                                  thrust::device_vector<point_t>& points_a,
                                  thrust::device_vector<point_t>& points_b,
                                  const std::vector<uint32_t>& max_hit_list) {
    Stopwatch sw;
    double total_time = 0;

    sw.start();
    auto n_points_a = points_a.size();
    auto n_points_b = points_b.size();
    const auto mbr_a = CalculateMbr(stream, points_a.begin(), points_a.end());
    const auto mbr_b = CalculateMbr(stream, points_b.begin(), points_b.end());

    HdBounds<COORD_T, N_DIMS> hd_bounds(mbr_a);
    auto hd_lb = hd_bounds.GetLowerBound(mbr_b);
    auto hd_ub = hd_bounds.GetUpperBound(mbr_b);
    COORD_T radius;
    // N.B., Do not shuffle A for better ray-coherence
    thrust::device_vector<point_t> points_b_shuffled = points_b;

    cmax2_.set(stream.cuda_stream(), hd_lb * hd_lb);
    thrust::shuffle(thrust::cuda::par.on(stream.cuda_stream()),
                    points_b_shuffled.begin(), points_b_shuffled.end(), g_);
    stats_.clear();
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
    sw.stop();
    total_time += sw.ms();

    // Sample points for a better initial HD
    {
      sw.start();
      uint32_t n_samples = ceil(points_a.size() * config_.sample_rate);
      auto sampled_point_ids_a =
          sampler_.Sample(stream.cuda_stream(), points_a.size(), n_samples);
      CalculateHDEarlyBreak(stream, points_a, points_b_shuffled,
                            sampled_point_ids_a);
      auto sampled_hd2 = cmax2_.get(stream.cuda_stream());
      radius = sqrt(sampled_hd2);
      sw.stop();
      stats_["NumSamples"] = n_samples;
      stats_["SampleTime"] = sw.ms();
      stats_["HD2AfterSampling"] = sampled_hd2;
      total_time += sw.ms();
    }

    sw.start();
    if (radius == 0) {
      radius = hd_lb;
    }
    if (radius == 0) {
      radius = hd_ub / 100;
    }
    CHECK_GT(radius, 0);
    COORD_T max_radius = hd_ub;

    stats_["HDLowerBound"] = hd_lb;
    stats_["HDUpperBound"] = hd_ub;
    stats_["InitRadius"] = radius;

    auto grid_size =
        grid_.CalculateGridResolution(mbr_b, n_points_b, config_.n_points_cell);
    grid_.Init(grid_size, mbr_b);
    grid_.Insert(stream, points_b);
    auto mbrs_b = grid_.GetCellMbrs(stream);
    stats_["Grid"] = grid_.GetStats();
    sw.stop();
    total_time += sw.ms();

    sw.start();
    auto mem_bytes = rt_engine_.EstimateMemoryUsageForAABB(
        mbrs_b.size(), config_.fast_build, config_.compact);
    buffer_.Init(mem_bytes * 1.2);
    buffer_.Clear();
    auto gas_handle = BuildBVH(stream, mbrs_b, radius);
    stream.Sync();
    sw.stop();

    stats_["BVHBuildTime"] = sw.ms();
    stats_["BVHMemoryKB"] = mem_bytes / 1024;

    thrust::device_vector<uint32_t> morton_codes;
    SharedValue<uint32_t> iter_hits;
    int iter = 0;
#ifdef PROFILING
    struct hdr_histogram* histogram;
    // Initialise the histogram
    hdr_init(1,                     // Minimum value
             (int64_t) n_points_b,  // Maximum value
             3,                     // Number of significant figures
             &histogram);           // Pointer to initialise
#endif

    uint32_t in_size = n_points_a;

    while (in_size > 0) {
      iter++;
      auto& json_iter = stats_["Iter" + std::to_string(iter)];
      auto* p_points_a = thrust::raw_pointer_cast(points_a.data());

      if (config_.sort_rays) {
        sw.start();
        morton_codes.resize(in_size);
        thrust::transform(thrust::cuda::par.on(stream.cuda_stream()),
                          in_queue_.data(), in_queue_.data() + in_size,
                          morton_codes.begin(), [=] __device__(uint32_t pid) {
                            const auto& p = p_points_a[pid];
                            return details::morton_code(p);
                          });
        thrust::sort_by_key(thrust::cuda::par.on(stream.cuda_stream()),
                            morton_codes.begin(), morton_codes.end(),
                            in_queue_.data());
        stream.Sync();
        sw.stop();
        json_iter["SortRaysTime"] = sw.ms();
        total_time += sw.ms();
      }

      auto cmax2 = cmax2_.get(stream.cuda_stream());
      auto min_time = std::numeric_limits<double>::max();
      nlohmann::json best_json_iter;

      for (auto max_hit : max_hit_list) {
        auto curr_json_iter = json_iter;
        double compute_time = 0;

        sw.start();
        iter_hits.set(stream.cuda_stream(), 0);
        term_queue_.Clear(stream.cuda_stream());
        out_queue_.Clear(stream.cuda_stream());
        // restore cmax2 for different max hit
        cmax2_.set(stream.cuda_stream(), cmax2);

        details::LaunchParamsNNCompress<COORD_T, N_DIMS> params;

        params.in_queue = ArrayView<uint32_t>(in_queue_.data(), in_size);
        params.term_queue = term_queue_.DeviceObject();
        params.miss_queue = out_queue_.DeviceObject();
        params.points_a = points_a;
        params.points_b = points_b;
        params.handle = gas_handle;
        params.cmax2 = cmax2_.data();
        params.radius = radius;
        params.mbrs_b = mbrs_b;
        params.prefix_sum = grid_.get_prefix_sum();
        params.point_b_ids = grid_.get_point_ids();
        params.n_hits = iter_hits.data();
#ifdef PROFILING
        hits_counters_.resize(in_size, 0);
        params.hits_counters = thrust::raw_pointer_cast(hits_counters_.data());
#else
        params.hits_counters = nullptr;
#endif
        params.max_hit = max_hit;

        details::ModuleIdentifier mod_nn = details::NUM_MODULE_IDENTIFIERS;

        if (typeid(COORD_T) == typeid(float)) {
          if (N_DIMS == 2) {
            mod_nn = details::MODULE_ID_FLOAT_NN_COMPRESS_2D;
          } else if (N_DIMS == 3) {
            mod_nn = details::MODULE_ID_FLOAT_NN_COMPRESS_3D;
          }
        } else if (typeid(COORD_T) == typeid(double)) {
          if (N_DIMS == 2) {
            mod_nn = details::MODULE_ID_DOUBLE_NN_COMPRESS_2D;
          } else if (N_DIMS == 3) {
            mod_nn = details::MODULE_ID_DOUBLE_NN_COMPRESS_3D;
          }
        }

        dim3 dims{in_size, 1, 1};

        rt_engine_.CopyLaunchParams(stream.cuda_stream(), params);
        rt_engine_.Render(stream.cuda_stream(), mod_nn, dims);
        auto miss_size = out_queue_.size(stream.cuda_stream());
        auto term_size = term_queue_.size(stream.cuda_stream());
        ArrayView<uint32_t> eb_point_a_ids(term_queue_.data(), term_size);
        sw.stop();

        compute_time += sw.ms();

#ifdef PROFILING
        thrust::host_vector<uint32_t> h_hits_counters = hits_counters_;

        for (auto val : h_hits_counters) {
          if (val > 0)
            hdr_record_value(histogram,  // Histogram to record to
                             val);       // Value to record
        }
        json_iter["HitsHistogram"] = DumpHistogram(histogram);
        hdr_reset(histogram);
#endif

        curr_json_iter["NumInputPoints"] = in_size;
        curr_json_iter["NumOutputPoints"] = miss_size;
        curr_json_iter["NumTermPoints"] = term_size;
        curr_json_iter["MaxHit"] = max_hit;
        curr_json_iter["CMax2"] = cmax2_.get(stream.cuda_stream());
        curr_json_iter["RTTime"] = sw.ms();
        curr_json_iter["Hits"] = iter_hits.get(stream.cuda_stream());

        if (!eb_point_a_ids.empty()) {
          sw.start();
          thrust::shuffle(thrust::cuda::par.on(stream.cuda_stream()),
                          eb_point_a_ids.begin(), eb_point_a_ids.end(), g_);
          auto eb_compared_pairs = CalculateHDEarlyBreak(
              stream, points_a, points_b_shuffled, eb_point_a_ids);
          sw.stop();
          curr_json_iter["ComparedPairs"] = eb_compared_pairs;
          curr_json_iter["EBTime"] = sw.ms();
          compute_time += sw.ms();
        } else {
          curr_json_iter["ComparedPairs"] = 0;
          curr_json_iter["EBTime"] = 0;
        }

        if (compute_time < min_time) {
          min_time = compute_time;
          best_json_iter = curr_json_iter;
        }
      }
      json_iter = best_json_iter;
      total_time += min_time;

      in_queue_.Clear(stream.cuda_stream());
      in_queue_.Swap(out_queue_);
      in_size = in_queue_.size(stream.cuda_stream());

      if (in_size > 0) {
        sw.start();
        if (radius < max_radius) {
          radius *= config_.radius_step;
          radius = std::min(radius, max_radius);
        }
        if (config_.rebuild_bvh) {
          buffer_.Clear();
          gas_handle = BuildBVH(stream, mbrs_b, radius);
        } else {
          gas_handle = UpdateBVH(stream, gas_handle, mbrs_b, radius);
        }
        stream.Sync();
        sw.stop();
        json_iter["AdjustBVHTime"] = sw.ms();
        total_time += sw.ms();
      }
    }

    stats_["TotalTime"] = total_time;
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

  OptixTraversableHandle BuildBVH(const Stream& stream, ArrayView<mbr_t> mbrs,
                                  COORD_T radius) {
    aabbs_.resize(mbrs.size());
    thrust::transform(thrust::cuda::par.on(stream.cuda_stream()), mbrs.begin(),
                      mbrs.end(), aabbs_.begin(),
                      [=] __device__(const mbr_t& mbr) {
                        return details::GetOptixAABB(mbr, radius);
                      });
    return rt_engine_.BuildAccelCustom(stream.cuda_stream(),
                                       ArrayView<OptixAabb>(aabbs_), buffer_,
                                       config_.fast_build, config_.compact);
  }

  OptixTraversableHandle UpdateBVH(const Stream& stream,
                                   OptixTraversableHandle handle,
                                   ArrayView<mbr_t> mbrs, COORD_T radius) {
    aabbs_.resize(mbrs.size());
    thrust::transform(thrust::cuda::par.on(stream.cuda_stream()), mbrs.begin(),
                      mbrs.end(), aabbs_.begin(),
                      [=] __device__(const mbr_t& mbr) {
                        return details::GetOptixAABB(mbr, radius);
                      });
    return rt_engine_.UpdateAccelCustom(stream.cuda_stream(), handle,
                                        ArrayView<OptixAabb>(aabbs_), buffer_,
                                        0, config_.fast_build, config_.compact);
  }

  uint64_t CalculateHDEarlyBreak(
      const Stream& stream, const thrust::device_vector<point_t>& points_a,
      const thrust::device_vector<point_t>& points_b,
      ArrayView<uint32_t> v_point_ids_a = ArrayView<uint32_t>()) {
    SharedValue<unsigned long long int> compared_pairs;
    auto* p_cmax2 = cmax2_.data();
    ArrayView<point_t> v_points_a(points_a);
    ArrayView<point_t> v_points_b(points_b);

    auto* p_compared_pairs = compared_pairs.data();

    uint32_t n_points_a = v_point_ids_a.size();

    if (n_points_a == 0) {
      n_points_a = points_a.size();
    }

    compared_pairs.set(stream.cuda_stream(), 0);

    LaunchKernel(stream, [=] __device__() {
      using BlockReduce = cub::BlockReduce<coord_t, MAX_BLOCK_SIZE>;
      __shared__ typename BlockReduce::TempStorage temp_storage;
      __shared__ bool early_break;
      __shared__ const point_t* point_a;
      uint64_t n_pairs = 0;

      for (auto i = blockIdx.x; i < n_points_a; i += gridDim.x) {
        auto size_b_roundup =
            div_round_up(v_points_b.size(), blockDim.x) * blockDim.x;
        coord_t cmin = std::numeric_limits<coord_t>::max();

        if (threadIdx.x == 0) {
          early_break = false;
          if (v_point_ids_a.empty()) {
            point_a = &v_points_a[i];
          } else {
            auto point_id_s = v_point_ids_a[i];
            point_a = &v_points_a[point_id_s];
          }
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
            if (cmin <= *p_cmax2) {
              early_break = true;
            }
          }
          __syncthreads();
        }

        __syncthreads();
        if (threadIdx.x == 0 && cmin != std::numeric_limits<coord_t>::max()) {
          atomicMax(p_cmax2, cmin);
        }
      }
      atomicAdd(p_compared_pairs, n_pairs);
    });
    return compared_pairs.get(stream.cuda_stream());
  }

 private:
  HausdorffDistanceRTConfig config_;
  nlohmann::json stats_;
  thrust::default_random_engine g_;
  Grid<COORD_T, N_DIMS> grid_;
  thrust::device_vector<OptixAabb> aabbs_;
  thrust::device_vector<uint32_t> hits_counters_;
  SharedValue<mbr_t> mbr_;
  Queue<uint32_t> in_queue_, out_queue_;
  Queue<uint32_t> term_queue_;
  ReusableBuffer buffer_;
  details::RTEngine rt_engine_;
  SharedValue<COORD_T> cmax2_;
  Sampler sampler_;
};
}  // namespace hd

#endif  // HAUSDORFF_DISTANCE_RT_H
