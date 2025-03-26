#ifndef HAUSDORFF_DISTANCE_RT_H
#define HAUSDORFF_DISTANCE_RT_H
#include <glog/logging.h>
#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <thrust/shuffle.h>

#include <cmath>
#include <iomanip>

#include "distance.h"
#include "grid.h"
#include "hdr/hdr_histogram.h"
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
  float sample_rate = 0.0001;
  float init_radius = 0;
  int grid_size = 1024;
  int max_samples = 100 * 1000;
  int max_hit = 1000;
  int max_reg_count = 0;
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
    sampler_.Init(hd_config.max_samples);
    g_ = thrust::default_random_engine(config_.seed);
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
    LOG(INFO) << "Primitives " << n_points_b << " BVH Mem " << mem_bytes / 1024
              << " KB";
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
    COORD_T max_radius = hd_ub;

    LOG(INFO) << "LB " << hd_lb << " UB " << hd_ub << " Init Radius " << radius;

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

      dim3 dims{1, 1, 1};
      Stopwatch sw;
      sw.start();
      dims.x = in_size;
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
      std::ofstream ofs;
      std::string path = "/tmp/iter_" + std::to_string(iter);
      FILE* file = fopen(path.c_str(), "w");  // Open file in write mode

      if (file == nullptr) {
        printf("Error opening file!\n");
        return 1;  // Exit with error
      }

      hdr_percentiles_print(histogram,
                            file,  // File to write to
                            3,     // Granularity of printed values
                            1.0,   // Multiplier for results
                            CSV);  // Format CLASSIC/CSV supported.
      hdr_reset(histogram);
      fclose(file);  // Close the file
#endif
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

  COORD_T CalculateDistanceCompress(const Stream& stream,
                                    thrust::device_vector<point_t>& points_a,
                                    thrust::device_vector<point_t>& points_b) {
    auto n_points_a = points_a.size();
    auto n_points_b = points_b.size();
    const auto mbr_a = CalculateMbr(stream, points_a.begin(), points_a.end());
    const auto mbr_b = CalculateMbr(stream, points_b.begin(), points_b.end());
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
    COORD_T max_radius = hd_ub;

    LOG(INFO) << "LB " << hd_lb << " UB " << hd_ub << " Init Radius " << radius;

    in_queue_.Init(n_points_a);
    term_queue_.Init(n_points_a);
    out_queue_.Init(n_points_a);
    in_queue_.Clear(stream.cuda_stream());
    out_queue_.Clear(stream.cuda_stream());
    sw.start();
    grid_.Init(config_.grid_size, mbr_b);
    grid_.Insert(stream, points_b);
    auto mbrs_b = grid_.GetCellMbrs(stream);
    stream.Sync();
    sw.stop();
    LOG(INFO) << "Grid time " << sw.ms() << " ms";

#ifdef PROFILING
    grid_.PrintHistogram();
#endif

    auto mem_bytes = rt_engine_.EstimateMemoryUsageForAABB(
        mbrs_b.size(), config_.fast_build, config_.compact);
    LOG(INFO) << "Primitives " << mbrs_b.size() << " BVH Mem "
              << mem_bytes / 1024 << " KB";

    buffer_.Init(mem_bytes * 1.2);
    buffer_.Clear();

    auto d_in_queue = in_queue_.DeviceObject();

    thrust::for_each(thrust::cuda::par.on(stream.cuda_stream()),
                     thrust::make_counting_iterator<uint32_t>(0),
                     thrust::make_counting_iterator<uint32_t>(points_a.size()),
                     [=] __device__(uint32_t point_id) mutable {
                       d_in_queue.Append(point_id);
                     });

    uint32_t in_size = n_points_a;

    auto gas_handle = BuildBVH(stream, mbrs_b, radius);
    stream.Sync();

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
      details::LaunchParamsNNCompress<COORD_T, N_DIMS> params;
      ArrayView<uint32_t> in(in_queue_.data(), in_size);

      params.in_queue = in;
      params.miss_queue = out_queue_.DeviceObject();
      params.points_a = thrust::raw_pointer_cast(points_a.data());
      params.points_b = thrust::raw_pointer_cast(points_b.data());
      params.handle = gas_handle;
      params.cmax2 = cmax2_.data();
      params.radius = radius;
      params.mbrs_b = thrust::raw_pointer_cast(mbrs_b.data());
      params.prefix_sum = grid_.get_prefix_sum().data();
      params.point_b_ids = grid_.get_point_ids().data();

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
        }
      } else if (typeid(COORD_T) == typeid(double)) {
        if (N_DIMS == 2) {
        } else if (N_DIMS == 3) {
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

#ifdef PROFILING
      thrust::host_vector<uint32_t> h_hits_counters = hits_counters_;

      for (auto val : h_hits_counters) {
        if (val > 0)
          hdr_record_value(histogram,  // Histogram to record to
                           val);       // Value to record
      }
      std::ofstream ofs;
      std::string path = "/tmp/iter_" + std::to_string(iter);
      FILE* file = fopen(path.c_str(), "w");  // Open file in write mode

      if (file == nullptr) {
        printf("Error opening file!\n");
        return 1;  // Exit with error
      }

      hdr_percentiles_print(histogram,
                            file,  // File to write to
                            3,     // Granularity of printed values
                            1.0,   // Multiplier for results
                            CSV);  // Format CLASSIC/CSV supported.
      hdr_reset(histogram);
      fclose(file);  // Close the file
#endif
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
          gas_handle = BuildBVH(stream, mbrs_b, radius);
        } else {
          gas_handle = UpdateBVH(stream, gas_handle, mbrs_b, radius);
        }
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
    // Running EB on A sample of points A helps to set a good starting point of
    // cmax2, which helps RT method to early break
    sw.start();
    auto sampled_point_ids_a = sampler_.Sample(
        stream.cuda_stream(), points_a.size(),
        std::max(1u, (uint32_t) (points_a.size() * config_.sample_rate)));
    CalculateHDEarlyBreak(stream, points_a, points_b, sampled_point_ids_a);
    auto sampled_hd2 = cmax2_.get(stream.cuda_stream());
    sw.stop();
    VLOG(1) << "Sampled HD " << sqrt(sampled_hd2) << " time " << sw.ms()
            << " ms";

    if (hd_lb > 0) {
      radius = hd_lb;
    } else {
      radius = sqrt(sampled_hd2);
    }

    COORD_T max_radius = hd_ub;

    VLOG(1) << "LB " << hd_lb << " UB " << hd_ub << " Init Radius " << radius;

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

    auto gas_handle = BuildBVH(stream, points_b, radius);

    SharedValue<uint32_t> iter_hits;
    int iter = 0;
    uint64_t total_hits = 0;

    while (in_size > 0) {
      iter++;
      iter_hits.set(stream.cuda_stream(), 0);
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
          mod_nn = details::MODULE_ID_FLOAT_NN_3D;
        }
      } else if (typeid(COORD_T) == typeid(double)) {
        if (N_DIMS == 2) {
          mod_nn = details::MODULE_ID_DOUBLE_NN_2D;
        } else if (N_DIMS == 3) {
          mod_nn = details::MODULE_ID_DOUBLE_NN_3D;
        }
      }

      ArrayView<uint32_t> eb_point_a_ids;
      std::stringstream ss;

      ss << "Iter " << iter << " Radius " << radius << " In " << in_size;

      dim3 dims{in_size, 1, 1};
      sw.start();
      rt_engine_.CopyLaunchParams(stream.cuda_stream(), params);
      rt_engine_.Render(stream.cuda_stream(), mod_nn, dims);
      auto miss_size = out_queue_.size(stream.cuda_stream());
      auto term_size = term_queue_.size(stream.cuda_stream());
      eb_point_a_ids = ArrayView<uint32_t>(term_queue_.data(), term_size);
      auto n_hits = iter_hits.get(stream.cuda_stream());
      sw.stop();
      total_hits += n_hits;
      ss << " Term " << term_size << " Miss " << miss_size << " Iter Hits "
         << n_hits << " RT Time " << sw.ms() << " ms";

      VLOG(1) << "cmax2 " << cmax2_.get(stream.cuda_stream());

      if (!eb_point_a_ids.empty()) {
        sw.start();
        auto eb_compared_pairs =
            CalculateHDEarlyBreak(stream, points_a, points_b, eb_point_a_ids);
        VLOG(1) << "after EB cmax2 " << cmax2_.get(stream.cuda_stream());
        sw.stop();
        ss << " EB Time " << sw.ms() << " ms";
      }

      VLOG(1) << ss.str();
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
    VLOG(1) << "Total Hits " << total_hits;
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

    if (n_points_a < 1024 && points_b.size() > n_points_a && false) {
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
        __shared__ const point_t* point_a;

        for (auto i = blockIdx.x; i < n_points_a; i += gridDim.x) {
          auto size_b_roundup =
              div_round_up(v_points_b.size(), blockDim.x) * blockDim.x;
          coord_t cmin = std::numeric_limits<coord_t>::max();
          uint32_t n_pairs = 0;

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
  Grid<COORD_T, N_DIMS> grid_;
  thrust::device_vector<OptixAabb> aabbs_;
  thrust::device_vector<COORD_T> cmin2_;
  thrust::device_vector<uint32_t> hits_counters_;
  SharedValue<mbr_t> mbr_;
  OptixTraversableHandle gas_handle_;
  Queue<uint32_t> in_queue_, out_queue_;
  Queue<uint32_t> term_queue_;
  ReusableBuffer buffer_;
  details::RTEngine rt_engine_;
  SharedValue<COORD_T> cmax2_;
  SharedValue<uint32_t> compared_pairs_;
  Sampler sampler_;
};
}  // namespace hd

#endif  // HAUSDORFF_DISTANCE_RT_H
