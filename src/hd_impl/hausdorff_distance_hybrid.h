#ifndef HAUSDORFF_DISTANCE_HYBRID_H
#define HAUSDORFF_DISTANCE_HYBRID_H
#include <cooperative_groups.h>
#include <glog/logging.h>
#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <thrust/shuffle.h>

#include <cub/grid/grid_barrier.cuh>
#include <sstream>

#include "flags.h"
#include "geoms/distance.h"
#include "geoms/hd_bounds.h"
#include "geoms/mbr.h"
#include "hausdorff_distance.h"
#include "hd_impl/primitive_utils.h"
#include "index/uniform_grid.h"
#include "models/features.h"
#include "models/tree_ebonlythreshold_3d.h"
#include "models/tree_maxhit_3d.h"
#include "models/tree_numpointspercell_3d.h"
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

// #define PROFILING
namespace hd {

template <typename COORD_T, int N_DIMS>
class HausdorffDistanceHybrid : public HausdorffDistance<COORD_T, N_DIMS> {
  using base_t = HausdorffDistance<COORD_T, N_DIMS>;
  using coord_t = COORD_T;
  using point_t = typename base_t::point_t;
  using mbr_t = Mbr<COORD_T, N_DIMS>;
  using grid_t = UniformGrid<COORD_T, N_DIMS>;

 public:
  struct Config {
    const char* ptx_root;
    bool auto_tune = false;
    uint32_t hybrid_factor = 2048;
    int seed = 0;
    bool fast_build = false;
    bool compact = false;
    bool rebuild_bvh = false;
    float sample_rate = 0.001;
    int n_points_cell = 8;
    uint32_t sample_threshold = 500 * 1000;
    uint32_t eb_only_threshold = 0;
    int max_reg_count = 0;
    uint32_t max_hit = 0;
  };

  HausdorffDistanceHybrid() = default;

  explicit HausdorffDistanceHybrid(const Config& config) : config_(config) {
    auto rt_config = details::get_default_rt_config(config_.ptx_root);

    rt_config.max_reg_count = config_.max_reg_count;
    rt_engine_.Init(rt_config);
    sampler_ = Sampler(config_.seed);

    g_ = thrust::default_random_engine(config_.seed);
    CHECK_GT(config_.n_points_cell, 0);
    int device;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&dev_prop_, device);
  }

  void UpdateConfig(const Config& config) { config_ = config; }

  coord_t CalculateDistance(const Stream& stream,
                            thrust::device_vector<point_t>& points_a,
                            thrust::device_vector<point_t>& points_b) override {
    Stopwatch sw, sw_total, sw1;
    sw_total.start();
    sw1.start();
    auto& stats = this->stats_;
    auto n_points_a = points_a.size();
    auto n_points_b = points_b.size();
    const auto mbr_a =
        CalculateMbr(stream, points_a.begin(), points_a.end()).ToNonemptyMBR();
    const auto mbr_b =
        CalculateMbr(stream, points_b.begin(), points_b.end()).ToNonemptyMBR();

    HdBounds<COORD_T, N_DIMS> hd_bounds(mbr_a);
    auto hd_lb = hd_bounds.GetLowerBound(mbr_b);
    auto hd_ub = hd_bounds.GetUpperBound(mbr_b);

    stats.clear();
    stats["HDLowerBound"] = hd_lb;
    stats["HDUpperBound"] = hd_ub;

    if (hd_ub == 0) {
      return 0;
    }
    cmax2_.set(stream.cuda_stream(), hd_lb * hd_lb);

    auto sample_rate = get_sample_rate();
    auto n_points_cell = get_n_points_cell();

    // Sampling to find a good starting point of cmax2
    COORD_T radius = hd_lb;
    thrust::device_vector<point_t> points_b_shuffled;
    stats["SampleThreshold"] = config_.sample_threshold;

    auto max_active_blocks = CalculateHDEarlyBreak(stream, points_a, points_b,
                                                   ArrayView<uint32_t>(), true);

    if (n_points_b < config_.sample_threshold) {
      points_b_shuffled = points_b;
      thrust::shuffle(thrust::cuda::par.on(stream.cuda_stream()),
                      points_b_shuffled.begin(), points_b_shuffled.end(), g_);

      uint32_t n_samples = ceil(points_a.size() * sample_rate);
      thrust::device_vector<uint32_t> iter_counters(n_samples);
      sampler_.Init(n_samples);
      sw.start();
      auto sampled_point_ids_a =
          sampler_.Sample(stream.cuda_stream(), points_a.size(), n_samples);
      CalculateHDEarlyBreak(stream, points_a, points_b_shuffled,
                            sampled_point_ids_a, false,
                            thrust::raw_pointer_cast(iter_counters.data()));
      thrust::sort(thrust::cuda::par.on(stream.cuda_stream()),
                   iter_counters.begin(), iter_counters.end());
      COORD_T cmax2 = sqrt(cmax2_.get(stream.cuda_stream()));
      if (radius == 0) {
        radius = cmax2;
      }
      sw.stop();
      // thrust::host_vector<uint32_t> h_iter_counter = iter_counters;
      // for (int i = 0; i < h_iter_counter.size(); i++) {
      //   printf("%d, cnt %u\n", i, h_iter_counter[i]);
      // }

      uint32_t p50_iters = iter_counters[iter_counters.size() * 0.5];
      uint32_t p80_iters = iter_counters[iter_counters.size() * 0.8];
      uint32_t p95_iters = iter_counters[iter_counters.size() * 0.95];
      uint32_t p99_iters = iter_counters[iter_counters.size() * 0.99];
      uint32_t max_iters = iter_counters[iter_counters.size() - 1];

      stats["NumSamples"] = n_samples;
      stats["SampleTime"] = sw.ms();
      stats["HDAfterSampling"] = cmax2;
      stats["SamplingP50Iters"] = p50_iters;
      stats["SamplingP80Iters"] = p80_iters;
      stats["SamplingP95Iters"] = p95_iters;
      stats["SamplingP99Iters"] = p99_iters;
      stats["SamplingMaxIters"] = max_iters;

      // Does not worth use RT
      if (p80_iters < get_eb_only_threshold()) {
        sw.start();
        // CalculateHDEarlyBreakWarp(stream, points_a, points_b_shuffled);
        // iter_counters.reserve(n_points_a);
        // CalculateHDEarlyBreak(stream, points_a,
        // points_b_shuffled,ArrayView<uint32_t>(), false, false,
        //                     thrust::raw_pointer_cast(iter_counters.data()));
        CalculateHDEarlyBreak(stream, points_a, points_b_shuffled);
        auto hd = sqrt(cmax2_.get(stream.cuda_stream()));
        sw.stop();
        // thrust::sort(thrust::cuda::par.on(stream.cuda_stream()),
        //              iter_counters.begin(), iter_counters.end());
        // uint32_t p50_iters = iter_counters[iter_counters.size() * 0.5];
        // uint32_t p80_iters = iter_counters[iter_counters.size() * 0.8];
        // uint32_t p95_iters = iter_counters[iter_counters.size() * 0.95];
        // uint32_t p99_iters = iter_counters[iter_counters.size() * 0.99];
        // uint32_t max_iters = iter_counters[iter_counters.size() - 1];
        // LOG(INFO) << "p50 " << p50_iters << " p80 " << p80_iters << " p95 "
        // << p95_iters << " p99 " << p99_iters;

        // LOG(INFO) << "SW2 " << sw2.ms();
        sw_total.stop();
        stats["EBOnly"] = true;
        stats["EBTime"] = sw.ms();
        stats["Algorithm"] = "Hybrid";
        stats["Execution"] = "GPU";
        stats["ReportedTime"] = sw_total.ms();
        return hd;
      }
    }

    grid_t grid;
    auto grid_size =
        grid_t::CalculateGridResolution(mbr_b, n_points_b, n_points_cell);

    grid.Init(grid_size, mbr_b);
    grid.Insert(stream, points_b);
    thrust::device_vector<mbr_t> mbrs = grid.GetTightCellMbrs(stream, points_b);

    if (radius == 0) {
      radius = grid.GetCellDigonalLength();
    }

    // Build BVH
    sw.start();
    buffer_.Clear();
    auto mem_bytes = rt_engine_.EstimateMemoryUsageForAABB(
        mbrs.size(), config_.fast_build, config_.compact);
    buffer_.Init(mem_bytes * 1.2);
    auto gas_handle = BuildBVH(stream, mbrs, radius);
    stream.Sync();
    sw.stop();

    stats["Grid"] = grid.GetStats();
    stats["BVHBuildTime"] = sw.ms();
    stats["BVHMemoryKB"] = mem_bytes / 1024;
    stats["InitRadius"] = radius;

    int iter = 0;

#ifdef PROFILING
    hit_counters_.resize(n_points_a, 0);
    point_counters_.resize(n_points_a, 0);
    stats["Profiling"] = true;
#else
    stats["Profiling"] = false;
#endif

    in_queue_.Init(n_points_a);
    in_queue_.SetSequence(stream.cuda_stream(), n_points_a);
    term_queue_.Init(n_points_a);
    miss_queue_.Init(n_points_a);
    term_queue_.Clear(stream.cuda_stream());
    miss_queue_.Clear(stream.cuda_stream());
    stream.Sync();
    sw1.stop();
    VLOG(1) << "Stage1 " << sw1.ms();

    uint32_t in_size = in_queue_.size(stream.cuda_stream());
#ifdef PROFILING
    hdr_histogram* histogram;
    hdr_init(1,                      // Minimum value
             (int64_t) mbrs.size(),  // Maximum value
             3,                      // Number of significant figures
             &histogram);            // Pointer to initialise
    thrust::host_vector<uint32_t> h_hits_counter(n_points_a);
#endif
    double profiling_ms = 0;

    sw1.start();

    auto& json_iters = stats["Iterations"];

    while (in_size > 0) {
      iter++;
      json_iters.push_back(nlohmann::json());
      nlohmann::json& json_iter = json_iters.back();
      json_iter["Iteration"] = iter;
      sw.start();
      details::LaunchParamsNNUniformGrid<COORD_T, N_DIMS> params;

      params.in_queue = ArrayView<uint32_t>(in_queue_.data(), in_size);
      params.term_queue = term_queue_.DeviceObject();
      params.miss_queue = miss_queue_.DeviceObject();
      params.points_a = points_a;
      params.points_b = points_b;
      params.handle = gas_handle;
      params.cmax2 = cmax2_.data();
      params.radius = radius;
      params.mbrs_b = mbrs;
      params.prefix_sum = grid.get_prefix_sum();
      params.point_b_ids = grid.get_point_ids();
#ifdef PROFILING
      params.hit_counters = thrust::raw_pointer_cast(hit_counters_.data());
      params.point_counters = thrust::raw_pointer_cast(point_counters_.data());
#else
      params.hit_counters = nullptr;
      params.point_counters = nullptr;
#endif
      // Does not have enough parallelism, do not consider EB
      auto max_hit = get_max_hit();
      if (max_hit == 0 || in_size <= max_active_blocks) {
        max_hit = std::numeric_limits<uint32_t>::max();
      }
      params.max_hits = max_hit;
      params.prune = true;
      params.eb = true;

      auto prev_cmax2 = cmax2_.get(stream.cuda_stream());

      rt_engine_.CopyLaunchParams(stream.cuda_stream(), params);
      rt_engine_.Render(stream.cuda_stream(), getRTModule(false),
                        dim3{in_size, 1, 1});
      auto cmax2 = cmax2_.get(stream.cuda_stream());
      sw.stop();

      auto miss_size = miss_queue_.size(stream.cuda_stream());
      auto term_size = term_queue_.size(stream.cuda_stream());

      json_iter["NumInputPoints"] = in_size;
      json_iter["NumOutputPoints"] = miss_size;
      json_iter["NumTermPoints"] = term_size;
      json_iter["RTTime"] = sw.ms();
      json_iter["Radius"] = radius;
      sw1.stop();
      VLOG(1) << "Iter: " << iter << " radius: " << radius << std::fixed
              << std::setprecision(8) << " cmax2: " << cmax2
              << " cmax: " << sqrt(cmax2) << " in_size: " << in_size
              << " miss_size: " << miss_size << " term_size: " << term_size
              << " RT Time: " << sw.ms() << " ms" << " Elapsed: " << sw1.ms();

      sw.start();
#ifdef PROFILING
      h_hits_counter = hit_counters_;

      hdr_reset(histogram);
      for (uint32_t i = 0; i < in_size; i++) {
        hdr_record_value(histogram,           // Histogram to record to
                         h_hits_counter[i]);  // Value to record
      }

      hdr_percentiles_print(histogram,
                            stdout,  // File to write to
                            5,       // Granularity of printed values
                            1.0,     // Multiplier for results
                            CSV);    // Format CLASSIC/CSV supported.
      sw.stop();
      profiling_ms += sw.ms();
#endif

      json_iter["CMax2"] = cmax2;

      in_queue_.Clear(stream.cuda_stream());
      in_queue_.Swap(miss_queue_);

      uint32_t last_in_size = in_size;
      in_size = in_queue_.size(stream.cuda_stream());

      if (in_size > 0 && radius < hd_ub) {
        auto reduced_factor = (float) (last_in_size - in_size) / last_in_size;
        float expand_factors[] = {8, 4, 2, 1};

        for (auto expand_factor : expand_factors) {
          if (reduced_factor < 1 / expand_factor) {
            radius += expand_factor * grid.GetCellDigonalLength();
            break;
          }
        }

        radius = std::min(radius, hd_ub);
        sw.start();
        if (config_.rebuild_bvh) {
          buffer_.Clear();
          gas_handle = BuildBVH(stream, mbrs, radius);
        } else {
          gas_handle = UpdateBVH(stream, gas_handle, mbrs, radius);
        }
        stream.Sync();
        sw.stop();
        json_iter["AdjustBVHTime"] = sw.ms();
      }
    }
    auto cmax2 = cmax2_.get(stream.cuda_stream());
    sw1.stop();
    VLOG(1) << "Stage2 " << sw1.ms() - profiling_ms;

    auto term_size = term_queue_.size(stream.cuda_stream());
    if (term_size > 0) {
      sw.start();
      if (points_b_shuffled.empty()) {
        points_b_shuffled = points_b;
        thrust::shuffle(thrust::cuda::par.on(stream.cuda_stream()),
                        points_b_shuffled.begin(), points_b_shuffled.end(), g_);
      }

      if (term_size < dev_prop_.multiProcessorCount) {
        CalculateHDEarlyBreakGrid(
            stream, points_a, points_b_shuffled,
            ArrayView<uint32_t>(term_queue_.data(), term_size));
      } else {
        CalculateHDEarlyBreak(
            stream, points_a, points_b_shuffled,
            ArrayView<uint32_t>(term_queue_.data(), term_size));
      }

      cmax2 = cmax2_.get(stream.cuda_stream());
      sw.stop();
      VLOG(1) << "EB Time " << sw.ms();
      stats["EBTime"] = sw.ms();
    }

    sw_total.stop();

    stats["Algorithm"] = "Hybrid";
    stats["Execution"] = "GPU";
    stats["ReportedTime"] = sw_total.ms() - profiling_ms;

    return sqrt(cmax2);
  }

  OptixTraversableHandle BuildBVH(const Stream& stream, ArrayView<mbr_t> mbrs,
                                  COORD_T radius) {
    CHECK_GT(radius, 0);
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
    CHECK_GT(radius, 0);
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

  uint32_t CalculateHDEarlyBreakWarp(
      const Stream& stream, const thrust::device_vector<point_t>& points_a,
      const thrust::device_vector<point_t>& points_b,
      ArrayView<uint32_t> v_point_ids_a = ArrayView<uint32_t>(),
      bool dry_run = false, uint32_t* p_iter_counters = nullptr) {
    auto* p_cmax2 = cmax2_.data();
    ArrayView<point_t> v_points_a(points_a);
    ArrayView<point_t> v_points_b(points_b);
    SharedValue<uint32_t> finished_counter;
    auto* p_finished_counter = finished_counter.data();

    uint32_t n_points_a = v_point_ids_a.size();

    if (n_points_a == 0) {
      n_points_a = points_a.size();
    }
    finished_counter.set(stream.cuda_stream(), 0);

    auto kernel = [=] __device__() mutable {
      using WarpReduce = cub::WarpReduce<coord_t>;
      __shared__
          typename WarpReduce::TempStorage temp_storage[MAX_BLOCK_SIZE / 32];
      auto size_b_roundup = div_round_up(v_points_b.size(), 32) * 32;

      auto lane_id = threadIdx.x % 32;
      auto g_warp_id = TID_1D / 32;
      auto warp_id = threadIdx.x / 32;
      auto n_warps = TOTAL_THREADS_1D / 32;

      for (auto i = g_warp_id; i < n_points_a; i += n_warps) {
        uint32_t point_id_a;
        if (v_point_ids_a.empty()) {
          point_id_a = i;
        } else {
          point_id_a = v_point_ids_a[i];
        }

        const point_t& point_a = v_points_a[point_id_a];

        int n_iter = 0;
        auto thread_min = std::numeric_limits<coord_t>::max();
        bool early_break = false;

        for (auto j = lane_id; j < size_b_roundup; j += 32, n_iter++) {
          if (j < v_points_b.size()) {
            const auto& point_b = v_points_b[j];
            thread_min =
                std::min(thread_min, EuclideanDistance2(point_a, point_b));
          }

          if (n_iter % 32 == 0) {
            if (__any_sync(0xffffffff, thread_min <= *p_cmax2)) {
              early_break = true;
              break;
            }
          }
        }

        if (!early_break) {
          auto agg_cmin2 =
              WarpReduce(temp_storage[warp_id]).Reduce(thread_min, cub::Min());
          if (lane_id == 0 &&
              agg_cmin2 != std::numeric_limits<coord_t>::max()) {
            atomicMax(p_cmax2, agg_cmin2);
          }
        }
      }
    };

    auto launch_dims = GetKernelLaunchParams(kernel);

    if (!dry_run) {
      LaunchKernel(stream, kernel);
      LOG(INFO) << "NO EB " << finished_counter.get(stream.cuda_stream());
    }

    return launch_dims.x * dev_prop_.multiProcessorCount;
  }

  uint32_t CalculateHDEarlyBreak(
      const Stream& stream, const thrust::device_vector<point_t>& points_a,
      const thrust::device_vector<point_t>& points_b,
      ArrayView<uint32_t> v_point_ids_a = ArrayView<uint32_t>(),
      bool dry_run = false, uint32_t* p_iter_counters = nullptr) {
    auto* p_cmax2 = cmax2_.data();
    ArrayView<point_t> v_points_a(points_a);
    ArrayView<point_t> v_points_b(points_b);
    SharedValue<uint32_t> finished_counter;
    auto* p_finished_counter = finished_counter.data();

    uint32_t n_points_a = v_point_ids_a.size();

    if (n_points_a == 0) {
      n_points_a = points_a.size();
    }
    finished_counter.set(stream.cuda_stream(), 0);

    auto kernel = [=] __device__() mutable {
      using BlockReduce = cub::BlockReduce<coord_t, MAX_BLOCK_SIZE>;
      __shared__ typename BlockReduce::TempStorage temp_storage;
      __shared__ bool s_early_break;
      // __shared__ point_t point_a;
      auto size_b_roundup =
          div_round_up(v_points_b.size(), blockDim.x) * blockDim.x;

      for (auto i = blockIdx.x; i < n_points_a; i += gridDim.x) {
        uint32_t point_id_a;

        // if (threadIdx.x == 0) {
        //   early_break = false;
        if (v_point_ids_a.empty()) {
          point_id_a = i;
        } else {
          point_id_a = v_point_ids_a[i];
        }
        auto point_a = v_points_a[point_id_a];
        // }
        //
        // __syncthreads();
        bool early_break = false;
        int n_iter = 0;
        auto thread_min = std::numeric_limits<coord_t>::max();

        for (auto j = threadIdx.x; j < size_b_roundup && !early_break;
             j += blockDim.x, n_iter++) {
          if (j < v_points_b.size()) {
            const auto& point_b = v_points_b[j];
            thread_min =
                std::min(thread_min, EuclideanDistance2(point_a, point_b));
          }

          // Reduce the frequency of sync
          if (n_iter % 32 == 0) {
            auto agg_min =
                BlockReduce(temp_storage).Reduce(thread_min, cub::Min());

            if (threadIdx.x == 0) {
              if (agg_min <= *p_cmax2) {
                s_early_break = true;
              } else {
                s_early_break = false;
              }
            }
            __syncthreads();
            early_break = s_early_break;
          }
        }
        auto agg_cmin2 =
            BlockReduce(temp_storage).Reduce(thread_min, cub::Min());

        if (threadIdx.x == 0) {
          if (!early_break &&
              agg_cmin2 != std::numeric_limits<coord_t>::max()) {
            atomicMax(p_cmax2, agg_cmin2);
            auto prev_count = atomicAdd(p_finished_counter, 1);
          }

          if (p_iter_counters != nullptr) {
            p_iter_counters[i] = n_iter;
          }
        }
      }
    };

    auto launch_dims = GetKernelLaunchParams(kernel);

    if (!dry_run) {
      LaunchKernel(stream, kernel);
    }

    return launch_dims.x * dev_prop_.multiProcessorCount;
  }

  uint32_t CalculateHDEarlyBreakGrid(
      const Stream& stream, const thrust::device_vector<point_t>& points_a,
      const thrust::device_vector<point_t>& points_b,
      ArrayView<uint32_t> v_point_ids_a = ArrayView<uint32_t>(),
      bool dry_run = false) {
    auto* p_cmax2 = cmax2_.data();
    ArrayView<point_t> v_points_a(points_a);
    ArrayView<point_t> v_points_b(points_b);
    SharedValue<int> val_early_break;
    SharedValue<coord_t> val_cmin2;
    auto* p_early_break = val_early_break.data();
    auto* p_cmin2 = val_cmin2.data();

    uint32_t n_points_a = v_point_ids_a.size();

    if (n_points_a == 0) {
      n_points_a = points_a.size();
    }

    auto kernel = [=] __device__() mutable {
      using BlockReduce = cub::BlockReduce<coord_t, MAX_BLOCK_SIZE>;
      __shared__ typename BlockReduce::TempStorage temp_storage;
      __shared__ point_t point_a;
      namespace cg = cooperative_groups;

      for (auto i = 0; i < n_points_a; i++) {
        uint32_t point_id_a;

        if (threadIdx.x == 0) {
          if (v_point_ids_a.empty()) {
            point_id_a = i;
          } else {
            point_id_a = v_point_ids_a[i];
          }
          point_a = v_points_a[point_id_a];
        }

        if (TID_1D == 0) {
          *p_early_break = 0;
          *p_cmin2 = std::numeric_limits<coord_t>::max();
        }

        cg::this_grid().sync();
        int n_iter = 0;
        auto thread_min = std::numeric_limits<coord_t>::max();

        for (auto j = TID_1D; !*p_early_break && j < v_points_b.size();
             j += TOTAL_THREADS_1D, n_iter++) {
          const auto& point_b = v_points_b[j];
          auto d2 = EuclideanDistance2(point_a, point_b);
          thread_min = std::min(thread_min, d2);

          if (thread_min <= *p_cmax2) {
            *p_early_break = 1;
            break;
          }
        }
        auto agg_cmin2 =
            BlockReduce(temp_storage).Reduce(thread_min, cub::Min());

        if (threadIdx.x == 0 &&
            agg_cmin2 != std::numeric_limits<coord_t>::max()) {
          atomicMin(p_cmin2, agg_cmin2);
        }
        cg::this_grid().sync();
        if (TID_1D == 0) {
          auto cmin2 = *p_cmin2;
          if (!*p_early_break && cmin2 != std::numeric_limits<coord_t>::max()) {
            atomicMax(p_cmax2, cmin2);
          }
        }
      }
    };

    auto launch_dims = GetKernelLaunchParams(kernel);

    if (!dry_run) {
      LaunchCooperativeKernel(stream, kernel);
    }

    return launch_dims.x * dev_prop_.multiProcessorCount;
  }

 private:
  Config config_;
  thrust::default_random_engine g_;
  thrust::device_vector<OptixAabb> aabbs_;
  thrust::device_vector<uint32_t> hit_counters_, point_counters_;
  Queue<uint32_t> in_queue_, term_queue_, miss_queue_;
  ReusableBuffer buffer_;
  details::RTEngine rt_engine_;
  SharedValue<COORD_T> cmax2_;
  Sampler sampler_;
  cudaDeviceProp dev_prop_;

  double get_sample_rate() const {
    auto sample_rate = config_.sample_rate;

    if (config_.auto_tune) {
      const auto& json_input = RunningStats::instance().Get();
      // Features features(json_input, N_DIMS);
      // auto feature_vals = features.Serialize();

      // sample_rate = PredictSampleRate_3D(feature_vals.data());
      // sample_rate = std::max(0.00001f, sample_rate);
    }
    return sample_rate;
  }

  uint32_t get_eb_only_threshold() {
    auto eb_only_threshold = config_.eb_only_threshold;

    if (config_.auto_tune) {
      const auto& json_input = RunningStats::instance().Get("Input");
      Features features(N_DIMS);

      features.SetStaticFeatures(json_input);
      features.SetRuntimeFeatures(this->stats_);
      auto feature_vals = features.Serialize();
      eb_only_threshold = PredictEBOnlyThreshold_3D(feature_vals.data());
      eb_only_threshold = std::max(1u, eb_only_threshold);

      this->stats_["PredictedEBOnlyThreshold"] = eb_only_threshold;

      VLOG(1) << "Predicted eb_only_threshold " << eb_only_threshold;
    }
    return eb_only_threshold;
  }

  uint32_t get_n_points_cell() {
    auto n_points_cell = config_.n_points_cell;

    if (config_.auto_tune) {
      const auto& json_input = RunningStats::instance().Get("Input");
      Features features(N_DIMS);

      features.SetStaticFeatures(json_input);
      features.SetRuntimeFeatures(this->stats_);
      auto feature_vals = features.Serialize();
      n_points_cell = PredictNumPointsPerCell_3D(feature_vals.data());
      n_points_cell = std::max(1, n_points_cell);

      this->stats_["PredictedNPointsCell"] = n_points_cell;
      VLOG(1) << "Predicted n_points_cell " << n_points_cell;
    }
    return n_points_cell;
  }

  uint32_t get_max_hit() {
    auto max_hit = config_.max_hit;

    if (config_.auto_tune) {
      const auto& json_input = RunningStats::instance().Get("Input");
      Features features(N_DIMS);

      features.SetStaticFeatures(json_input);
      features.SetRuntimeFeatures(this->stats_);
      auto feature_vals = features.Serialize();
      max_hit = PredictMaxHit_3D(feature_vals.data());
      max_hit = std::max(1u, max_hit);

      this->stats_["PredictedMaxHit"] = max_hit;
      VLOG(1) << "Predicted max_hit " << max_hit;
    }
    return max_hit;
  }

  details::ModuleIdentifier getRTModule(bool overlap_test) {
    details::ModuleIdentifier mod = details::NUM_MODULE_IDENTIFIERS;

    if (typeid(COORD_T) == typeid(float)) {
      if (N_DIMS == 2) {
        if (overlap_test) {
          mod = details::MODULE_ID_FLOAT_OVERLAP_UNIFORM_GRID_2D;
        } else {
          mod = details::MODULE_ID_FLOAT_NN_UNIFORM_GRID_2D;
        }
      } else if (N_DIMS == 3) {
        if (overlap_test) {
          mod = details::MODULE_ID_FLOAT_OVERLAP_UNIFORM_GRID_3D;
        } else {
          mod = details::MODULE_ID_FLOAT_NN_UNIFORM_GRID_3D;
        }
      }
    } else if (typeid(COORD_T) == typeid(double)) {
      if (N_DIMS == 2) {
        if (overlap_test) {
          mod = details::MODULE_ID_DOUBLE_OVERLAP_UNIFORM_GRID_2D;
        } else {
          mod = details::MODULE_ID_DOUBLE_NN_UNIFORM_GRID_2D;
        }
      } else if (N_DIMS == 3) {
        if (overlap_test) {
          mod = details::MODULE_ID_DOUBLE_OVERLAP_UNIFORM_GRID_3D;
        } else {
          mod = details::MODULE_ID_DOUBLE_NN_UNIFORM_GRID_3D;
        }
      }
    }
    return mod;
  }
};
}  // namespace hd
#undef PROFILING
#endif  // HAUSDORFF_DISTANCE_HYBRID_H
