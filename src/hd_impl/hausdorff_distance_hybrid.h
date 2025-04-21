#ifndef HAUSDORFF_DISTANCE_HYBRID_H
#define HAUSDORFF_DISTANCE_HYBRID_H
#include <glog/logging.h>
#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <thrust/shuffle.h>

#include <sstream>

#include "distance.h"
#include "grid.h"
#include "hausdorff_distance.h"
#include "hd_bounds.h"
#include "hd_impl/primitive_utils.h"
#include "mbr.h"
#include "models/features.h"
#include "models/tree_maxhitinit_3d.h"
#include "models/tree_maxhitnext_3d.h"
#include "models/tree_samplerate_3d.h"
#include "rt/launch_parameters.h"
#include "rt/reusable_buffer.h"
#include "rt/rt_engine.h"
#include "sampler.h"
#include "utils/array_view.h"
#include "utils/derived_atomic_functions.h"
#include "utils/helpers.h"
#include "utils/queue.h"
#include "utils/shared_value.h"
#include "utils/stopwatch.h"
#include "utils/stream.h"
#include "utils/type_traits.h"
#include "utils/util.h"

namespace hd {

template <typename COORD_T, int N_DIMS>
class HausdorffDistanceHybrid : public HausdorffDistance<COORD_T, N_DIMS> {
  using base_t = HausdorffDistance<COORD_T, N_DIMS>;
  using coord_t = COORD_T;
  using point_t = typename base_t::point_t;
  using mbr_t = Mbr<COORD_T, N_DIMS>;
  using grid_t = Grid<COORD_T, N_DIMS>;

 public:
  struct Config {
    const char* ptx_root;
    bool auto_tune = false;
    int seed = 0;
    bool fast_build = false;
    bool compact = false;
    bool rebuild_bvh = false;
    float radius_step = 2;
    float sample_rate = 0.001;
    bool sort_rays = false;
    int n_points_cell = 8;
    int max_samples = 100 * 1000;
    int max_reg_count = 0;
    int max_hit = 128;
  };

  HausdorffDistanceHybrid() = default;

  explicit HausdorffDistanceHybrid(const Config& config) : config_(config) {
    auto rt_config = details::get_default_rt_config(config_.ptx_root);

    rt_config.max_reg_count = config_.max_reg_count;
    rt_engine_.Init(rt_config);
    sampler_ = Sampler(config_.seed);
    sampler_.Init(config_.max_samples);
    g_ = thrust::default_random_engine(config_.seed);
    CHECK_GT(config_.n_points_cell, 0);
  }

  void UpdateConfig(const Config& config) {
    config_ = config;

  }

  coord_t CalculateDistance(const Stream& stream,
                            thrust::device_vector<point_t>& points_a,
                            thrust::device_vector<point_t>& points_b) override {
    uint64_t compared_pairs = 0;
    auto& stats = this->stats_;
    auto n_points_a = points_a.size();
    auto n_points_b = points_b.size();
    const auto mbr_a = CalculateMbr(stream, points_a.begin(), points_a.end());
    const auto mbr_b = CalculateMbr(stream, points_b.begin(), points_b.end());

    HdBounds<COORD_T, N_DIMS> hd_bounds(mbr_a);
    auto hd_lb = hd_bounds.GetLowerBound(mbr_b);
    auto hd_ub = hd_bounds.GetUpperBound(mbr_b);

    cmax2_.set(stream.cuda_stream(), hd_lb * hd_lb);

    stats.clear();

    stats["Seed"] = config_.seed;
    stats["SortRays"] = config_.sort_rays;
    stats["FastBuildBVH"] = config_.fast_build;
    stats["RebuildBVH"] = config_.rebuild_bvh;
    stats["RadiusStep"] = config_.radius_step;
    stats["NumPointsPerCell"] = config_.n_points_cell;

    Stopwatch sw;
    thrust::device_vector<point_t> points_b_shuffled = points_b;
    COORD_T radius;

    thrust::shuffle(thrust::cuda::par.on(stream.cuda_stream()),
                    points_b_shuffled.begin(), points_b_shuffled.end(), g_);

    // Sample points for a better initial HD
    {
      sw.start();
      auto sample_rate = get_sample_rate();
      uint32_t n_samples = ceil(points_a.size() * sample_rate);
      auto sampled_point_ids_a =
          sampler_.Sample(stream.cuda_stream(), points_a.size(), n_samples);
      compared_pairs += CalculateHDEarlyBreak(
          stream, points_a, points_b_shuffled, sampled_point_ids_a);
      auto sampled_hd2 = cmax2_.get(stream.cuda_stream());
      radius = sqrt(sampled_hd2);
      sw.stop();
      stats["SampleRate"] = sample_rate;
      stats["NumSamples"] = n_samples;
      stats["SampleTime"] = sw.ms();
      stats["HD2AfterSampling"] = sampled_hd2;
    }
    if (radius == 0) {
      radius = hd_lb;
    }
    if (radius == 0) {
      radius = hd_ub / 100;
    }
    CHECK_GT(radius, 0);

    COORD_T max_radius = hd_ub;

    stats["HDLowerBound"] = hd_lb;
    stats["HDUpperBound"] = hd_ub;
    stats["InitRadius"] = radius;

    in_queue_.Init(n_points_a);
    term_queue_.Init(n_points_a);
    out_queue_.Init(n_points_a);
    in_queue_.Clear(stream.cuda_stream());
    out_queue_.Clear(stream.cuda_stream());

    LOG(INFO) << "n_points/cell " << config_.n_points_cell;
    auto grid_size =
        grid_.CalculateGridResolution(mbr_b, n_points_b, config_.n_points_cell);

    grid_.Init(grid_size, mbr_b);
    grid_.Insert(stream, points_b);
    auto mbrs_b = grid_.GetCellMbrs(stream);

    stats["Grid"] = grid_.GetStats();

    // Build BVH
    sw.start();
    buffer_.Clear();
    auto mem_bytes = rt_engine_.EstimateMemoryUsageForAABB(
        mbrs_b.size(), config_.fast_build, config_.compact);
    buffer_.Init(mem_bytes * 1.2);
    auto gas_handle = BuildBVH(stream, mbrs_b, radius);
    stream.Sync();
    sw.stop();

    stats["BVHBuildTime"] = sw.ms();
    stats["BVHMemoryKB"] = mem_bytes / 1024;

    int iter = 0;
    uint32_t in_size = n_points_a;
    thrust::device_vector<uint32_t> morton_codes;
    auto max_hit = get_max_hit_init();

#ifdef PROFILING
    struct hdr_histogram* histogram;
    // Initialise the histogram
    hdr_init(1,                     // Minimum value
             (int64_t) n_points_b,  // Maximum value
             3,                     // Number of significant figures
             &histogram);           // Pointer to initialise
    stats["Profiling"] = true;
#else
    stats["Profiling"] = false;
#endif

    in_queue_.SetSequence(stream.cuda_stream(), in_size);

    while (in_size > 0) {
      iter++;
      auto& json_iter = stats["Iter" + std::to_string(iter)];

      if (config_.sort_rays) {
        auto* p_points_a = thrust::raw_pointer_cast(points_a.data());
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

      iter_hits_.set(stream.cuda_stream(), 0);
      term_queue_.Clear(stream.cuda_stream());

      sw.start();
      details::ModuleIdentifier mod_nn = details::NUM_MODULE_IDENTIFIERS;
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
      params.n_hits = iter_hits_.data();
#ifdef PROFILING
      hits_counters_.resize(in_size, 0);
      params.hits_counters = thrust::raw_pointer_cast(hits_counters_.data());
#else
      params.hits_counters = nullptr;
#endif
      params.max_hit = max_hit;

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
      rt_engine_.CopyLaunchParams(stream.cuda_stream(), params);
      rt_engine_.Render(stream.cuda_stream(), mod_nn, dim3{in_size, 1, 1});
      auto miss_size = out_queue_.size(stream.cuda_stream());
      auto term_size = term_queue_.size(stream.cuda_stream());
      sw.stop();

      json_iter["MaxHit"] = max_hit;
      json_iter["NumInputPoints"] = in_size;
      json_iter["NumOutputPoints"] = miss_size;
      json_iter["NumTermPoints"] = term_size;
      json_iter["CMax2"] = cmax2_.get(stream.cuda_stream());
      json_iter["RTTime"] = sw.ms();
      json_iter["Hits"] = iter_hits_.get(stream.cuda_stream());
      compared_pairs += json_iter["Hits"].template get<uint32_t>();

      ArrayView<uint32_t> eb_point_a_ids(term_queue_.data(), term_size);

      if (!eb_point_a_ids.empty()) {
        sw.start();
        thrust::shuffle(thrust::cuda::par.on(stream.cuda_stream()),
                        eb_point_a_ids.begin(), eb_point_a_ids.end(), g_);
        auto iter_compared_pairs = CalculateHDEarlyBreak(
            stream, points_a, points_b_shuffled, eb_point_a_ids);
        sw.stop();
        json_iter["ComparedPairs"] = iter_compared_pairs;
        json_iter["EBTime"] = sw.ms();
        compared_pairs += iter_compared_pairs;
      } else {
        json_iter["ComparedPairs"] = 0;
        json_iter["EBTime"] = 0;
      }

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
      max_hit = get_max_hit_next(json_iter);
    }
    auto cmax2 = cmax2_.get(stream.cuda_stream());

    stats["Algorithm"] = "Hybrid";
    stats["Execution"] = "GPU";
    stats["ComparedPairs"] = compared_pairs;

    return sqrt(cmax2);
  }

  coord_t CalculateDistance(const Stream& stream,
                            thrust::device_vector<point_t>& points_a,
                            thrust::device_vector<point_t>& points_b,
                            const std::vector<uint32_t>& max_hit_list) {
    Stopwatch sw;
    double total_time = 0;
    sw.start();
    uint64_t compared_pairs = 0;
    auto& stats = this->stats_;
    auto n_points_a = points_a.size();
    auto n_points_b = points_b.size();
    const auto mbr_a = CalculateMbr(stream, points_a.begin(), points_a.end());
    const auto mbr_b = CalculateMbr(stream, points_b.begin(), points_b.end());

    HdBounds<COORD_T, N_DIMS> hd_bounds(mbr_a);
    auto hd_lb = hd_bounds.GetLowerBound(mbr_b);
    auto hd_ub = hd_bounds.GetUpperBound(mbr_b);

    cmax2_.set(stream.cuda_stream(), hd_lb * hd_lb);

    stats.clear();

    stats["Seed"] = config_.seed;
    stats["SortRays"] = config_.sort_rays;
    stats["FastBuildBVH"] = config_.fast_build;
    stats["RebuildBVH"] = config_.rebuild_bvh;
    stats["RadiusStep"] = config_.radius_step;
    stats["NumPointsPerCell"] = config_.n_points_cell;

    thrust::device_vector<point_t> points_b_shuffled = points_b;
    COORD_T radius;

    thrust::shuffle(thrust::cuda::par.on(stream.cuda_stream()),
                    points_b_shuffled.begin(), points_b_shuffled.end(), g_);
    sw.stop();
    total_time += sw.ms();
    // Sample points for a better initial HD
    {
      sw.start();
      auto sample_rate = get_sample_rate();
      uint32_t n_samples = ceil(points_a.size() * sample_rate);
      auto sampled_point_ids_a =
          sampler_.Sample(stream.cuda_stream(), points_a.size(), n_samples);
      compared_pairs += CalculateHDEarlyBreak(
          stream, points_a, points_b_shuffled, sampled_point_ids_a);
      auto sampled_hd2 = cmax2_.get(stream.cuda_stream());
      radius = sqrt(sampled_hd2);
      sw.stop();
      stats["SampleRate"] = sample_rate;
      stats["NumSamples"] = n_samples;
      stats["SampleTime"] = sw.ms();
      stats["HD2AfterSampling"] = sampled_hd2;
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

    stats["HDLowerBound"] = hd_lb;
    stats["HDUpperBound"] = hd_ub;
    stats["InitRadius"] = radius;

    in_queue_.Init(n_points_a);
    term_queue_.Init(n_points_a);
    out_queue_.Init(n_points_a);
    in_queue_.Clear(stream.cuda_stream());
    out_queue_.Clear(stream.cuda_stream());

    auto grid_size =
        grid_.CalculateGridResolution(mbr_b, n_points_b, config_.n_points_cell);

    grid_.Init(grid_size, mbr_b);
    grid_.Insert(stream, points_b);
    auto mbrs_b = grid_.GetCellMbrs(stream);

    stats["Grid"] = grid_.GetStats();
    sw.stop();
    total_time += sw.ms();

    // Build BVH
    sw.start();
    buffer_.Clear();
    auto mem_bytes = rt_engine_.EstimateMemoryUsageForAABB(
        mbrs_b.size(), config_.fast_build, config_.compact);
    buffer_.Init(mem_bytes * 1.2);
    auto gas_handle = BuildBVH(stream, mbrs_b, radius);
    stream.Sync();
    sw.stop();

    stats["BVHBuildTime"] = sw.ms();
    stats["BVHMemoryKB"] = mem_bytes / 1024;
    total_time += sw.ms();

    sw.start();
    int iter = 0;
    uint32_t in_size = n_points_a;
    thrust::device_vector<uint32_t> morton_codes;

#ifdef PROFILING
    struct hdr_histogram* histogram;
    // Initialise the histogram
    hdr_init(1,                     // Minimum value
             (int64_t) n_points_b,  // Maximum value
             3,                     // Number of significant figures
             &histogram);           // Pointer to initialise
    stats["Profiling"] = true;
#else
    stats["Profiling"] = false;
#endif

    in_queue_.SetSequence(stream.cuda_stream(), in_size);
    stream.Sync();
    sw.stop();
    total_time += sw.ms();

    while (in_size > 0) {
      iter++;
      auto& json_iter = stats["Iter" + std::to_string(iter)];

      if (config_.sort_rays) {
        auto* p_points_a = thrust::raw_pointer_cast(points_a.data());
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
        double iter_time = 0;

        sw.start();
        iter_hits_.set(stream.cuda_stream(), 0);
        term_queue_.Clear(stream.cuda_stream());
        out_queue_.Clear(stream.cuda_stream());
        // restore cmax2 for different max hit
        cmax2_.set(stream.cuda_stream(), cmax2);

        details::ModuleIdentifier mod_nn = details::NUM_MODULE_IDENTIFIERS;
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
        params.n_hits = iter_hits_.data();
#ifdef PROFILING
        hits_counters_.resize(in_size, 0);
        params.hits_counters = thrust::raw_pointer_cast(hits_counters_.data());
#else
        params.hits_counters = nullptr;
#endif
        params.max_hit = max_hit;

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
        rt_engine_.CopyLaunchParams(stream.cuda_stream(), params);
        rt_engine_.Render(stream.cuda_stream(), mod_nn, dim3{in_size, 1, 1});
        auto miss_size = out_queue_.size(stream.cuda_stream());
        auto term_size = term_queue_.size(stream.cuda_stream());
        sw.stop();
        iter_time += sw.ms();

        json_iter["MaxHit"] = max_hit;
        json_iter["NumInputPoints"] = in_size;
        json_iter["NumOutputPoints"] = miss_size;
        json_iter["NumTermPoints"] = term_size;
        json_iter["CMax2"] = cmax2_.get(stream.cuda_stream());
        json_iter["RTTime"] = sw.ms();
        json_iter["Hits"] = iter_hits_.get(stream.cuda_stream());
        compared_pairs += json_iter["Hits"].template get<uint32_t>();

        ArrayView<uint32_t> eb_point_a_ids(term_queue_.data(), term_size);

        if (!eb_point_a_ids.empty()) {
          sw.start();
          thrust::shuffle(thrust::cuda::par.on(stream.cuda_stream()),
                          eb_point_a_ids.begin(), eb_point_a_ids.end(), g_);
          auto iter_compared_pairs = CalculateHDEarlyBreak(
              stream, points_a, points_b_shuffled, eb_point_a_ids);
          sw.stop();
          json_iter["ComparedPairs"] = iter_compared_pairs;
          json_iter["EBTime"] = sw.ms();
          compared_pairs += iter_compared_pairs;
          iter_time += sw.ms();
        } else {
          json_iter["ComparedPairs"] = 0;
          json_iter["EBTime"] = 0;
        }

        if (iter_time < min_time) {
          min_time = iter_time;
          best_json_iter = json_iter;
        }

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
      }
      total_time += min_time;

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
        total_time += sw.ms();
      }
    }
    auto cmax2 = cmax2_.get(stream.cuda_stream());

    stats["Algorithm"] = "Hybrid";
    stats["Execution"] = "GPU";
    stats["ComparedPairs"] = compared_pairs;
    stats["TotalTime"] = total_time;

    return sqrt(cmax2);
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
  Config config_;
  thrust::default_random_engine g_;
  grid_t grid_;
  thrust::device_vector<OptixAabb> aabbs_;
  thrust::device_vector<uint32_t> hits_counters_;
  SharedValue<mbr_t> mbr_;
  Queue<uint32_t> in_queue_, term_queue_, out_queue_;
  ReusableBuffer buffer_;
  details::RTEngine rt_engine_;
  SharedValue<COORD_T> cmax2_;
  Sampler sampler_;
  SharedValue<uint32_t> iter_hits_;

  double get_sample_rate() const {
    auto sample_rate = config_.sample_rate;

    if (config_.auto_tune) {
      const auto& json_input = RunningStats::instance().Get("Input");
      FeaturesStatic<N_DIMS, 8> features(json_input);
      auto feature_vals = features.Serialize();

      if (N_DIMS == 2) {
        // TODO
      } else if (N_DIMS == 3) {
        sample_rate = PredictSampleRate_3D(feature_vals.data());
      }
    }
    return sample_rate;
  }

  uint32_t get_n_points_cell() const {
    auto n_points_cell = config_.n_points_cell;

    if (config_.auto_tune) {
      const auto& json_input = RunningStats::instance().Get("Input");
      FeaturesStatic<N_DIMS, 8> features(json_input);
      auto feature_vals = features.Serialize();

      if (N_DIMS == 2) {
        // TODO
      } else if (N_DIMS == 3) {
        n_points_cell = PredictNumPointsPerCell_3D(feature_vals.data());
      }
    }
    return n_points_cell;
  }

  uint32_t get_max_hit_init() const {
    auto max_hit = config_.max_hit;

    if (config_.auto_tune) {
      const auto& json_input = RunningStats::instance().Get("Input");
      FeaturesMaxHitInit<N_DIMS> features_max_hit_init(json_input);

      features_max_hit_init.UpdateRunningInfo(this->stats_);

      auto features_vals = features_max_hit_init.Serialize();

      if (N_DIMS == 2) {
        // TODO
      } else if (N_DIMS == 3) {
        max_hit = PredictMaxHitInit_3D(features_vals.data());
      }
    }

    return max_hit;
  }

  uint32_t get_max_hit_next(const nlohmann::json& json_iter) const {
    auto max_hit = config_.max_hit;

    if (config_.auto_tune) {
      const auto& json_input = RunningStats::instance().Get("Input");
      FeaturesMaxHitNext<N_DIMS> features_max_hit_next(json_input);

      features_max_hit_next.UpdateRunningInfo(json_iter);

      auto features_vals = features_max_hit_next.Serialize();

      if (N_DIMS == 2) {
        // TODO
      } else if (N_DIMS == 3) {
        max_hit = PredictMaxHitNext_3D(features_vals.data());
        LOG(INFO) << "max_hit " << max_hit;
      }
    }

    return max_hit;
  }
};
}  // namespace hd

#endif  // HAUSDORFF_DISTANCE_HYBRID_H
