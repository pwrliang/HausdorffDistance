#ifndef HAUSDORFF_DISTANCE_HYBRID_H
#define HAUSDORFF_DISTANCE_HYBRID_H
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
#include "index/uniform_grid.h"
#include "models/features.h"
#include "models/tree_numpointspercell_3d.h"
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
    uint32_t max_samples = 100 * 1000;
    int max_reg_count = 0;
    float max_hit_ratio = 0;
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
    int device;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&dev_prop_, device);
  }

  void UpdateConfig(const Config& config) { config_ = config; }

  coord_t CalculateDistance(const Stream& stream,
                            thrust::device_vector<point_t>& points_a,
                            thrust::device_vector<point_t>& points_b) override {
    Stopwatch sw, sw_total;
    sw_total.start();
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

    if (hd_ub == 0) {
      return 0;
    }

    cmax2_.set(stream.cuda_stream(), hd_lb * hd_lb);

    stats.clear();

    config_.sample_rate = get_sample_rate();
    config_.n_points_cell = get_n_points_cell();

    if (config_.auto_tune) {
      VLOG(1) << "Sample Rate " << config_.sample_rate;
      VLOG(1) << "N Points Cell " << config_.n_points_cell;
    }

    stats["SampleRate"] = config_.sample_rate;
    stats["Seed"] = config_.seed;
    stats["FastBuildBVH"] = config_.fast_build;
    stats["RebuildBVH"] = config_.rebuild_bvh;
    stats["NumPointsPerCell"] = config_.n_points_cell;

    thrust::device_vector<point_t> points_b_shuffled = points_b;
    thrust::shuffle(thrust::cuda::par.on(stream.cuda_stream()),
                    points_b_shuffled.begin(), points_b_shuffled.end(), g_);

    COORD_T radius;
    uint64_t limit_kcycles;
    int block_size;
    // Sampling to find a good starting point of cmax2
    COORD_T sampled_hd2;
    {
      sw.start();
      uint32_t n_samples = ceil(points_a.size() * config_.sample_rate);
      n_samples = std::min(n_samples, config_.max_samples);
      thrust::device_vector<uint32_t> cycles(n_samples);
      thrust::device_vector<uint32_t> iters(n_samples);
      auto sampled_point_ids_a =
          sampler_.Sample(stream.cuda_stream(), points_a.size(), n_samples);
      block_size = CalculateHDEarlyBreak(
          stream, points_a, points_b_shuffled, sampled_point_ids_a,
          thrust::raw_pointer_cast(cycles.data()),
          thrust::raw_pointer_cast(iters.data()));
      sampled_hd2 = cmax2_.get(stream.cuda_stream());
      sw.stop();
      radius = sqrt(sampled_hd2);
      stats["NumSamples"] = n_samples;
      stats["SampleTime"] = sw.ms();
      stats["HD2AfterSampling"] = sampled_hd2;

      thrust::sort(thrust::cuda::par.on(stream.cuda_stream()), iters.begin(),
                   iters.end());
      uint32_t median_iters = iters[iters.size() / 2];

      auto parallelism = dev_prop_.multiProcessorCount * block_size;
      auto hybrid_threshold =
          ceil((float) parallelism / n_points_a * config_.hybrid_factor);
      stats["MedianEBIters"] = median_iters;
      stats["HybridThreshold"] = hybrid_threshold;

      if (median_iters < hybrid_threshold) {
        CalculateHDEarlyBreak(stream, points_a, points_b_shuffled);
        auto hd = cmax2_.get(stream.cuda_stream());
        sw_total.stop();
        stats["Algorithm"] = "Hybrid";
        stats["Execution"] = "GPU";
        stats["ReportedTime"] = sw_total.ms();
        return sqrt(hd);
      }

      thrust::sort(thrust::cuda::par.on(stream.cuda_stream()), cycles.begin(),
                   cycles.end());

      auto total_cycles = thrust::reduce(
          thrust::cuda::par.on(stream.cuda_stream()), cycles.begin(),
          cycles.end(), 0ul, thrust::plus<uint64_t>());

      auto min_kcycles =
          thrust::reduce(thrust::cuda::par.on(stream.cuda_stream()),
                         cycles.begin(), cycles.end(),
                         std::numeric_limits<uint32_t>::max(),
                         thrust::minimum<uint32_t>()) /
          1000;
      auto max_kcycles =
          thrust::reduce(thrust::cuda::par.on(stream.cuda_stream()),
                         cycles.begin(), cycles.end(), 0,
                         thrust::maximum<uint32_t>()) /
          1000;
      auto avg_kcycles = total_cycles / n_samples / 1000;

      uint32_t median_kcycles = cycles[cycles.size() / 2];

      median_kcycles /= 1000;

      limit_kcycles = median_kcycles;
      VLOG(1) << "Min kcycles " << min_kcycles << " Max kcycles " << max_kcycles
              << " Avg kcycles " << avg_kcycles << " Median kcycles "
              << median_kcycles;
      VLOG(1) << "Limit kilo cycles " << limit_kcycles;
    }

    {
      auto grid_size =
       grid_.CalculateGridResolution(mbr_a, n_points_a, config_.n_points_cell);
    }

    if (radius == 0) {
      auto center_point_a = CalculateCenterPoint(stream, points_a);
      auto center_point_b = CalculateCenterPoint(stream, points_b);
      auto center_distance = EuclideanDistance2(center_point_a, center_point_b);
      radius = sqrt(center_distance) / 2;
    }

    if (radius == 0) {
      radius = hd_ub / 100;
    }
    CHECK_GT(radius, 0);

    auto grid_size =
        grid_.CalculateGridResolution(mbr_b, n_points_b, config_.n_points_cell);
    grid_.Init(grid_size, mbr_b);
    grid_.Insert(stream, points_b);
    auto mbrs_b = grid_.GetTightCellMbrs(stream, points_b);
    auto n_mbrs = mbrs_b.size();

    // TODO: Idea, use center points of grid cells, to calculate approximate
    // number of compares

    stats["Grid"] = grid_.GetStats();

    // Build BVH
    sw.start();
    buffer_.Clear();
    auto mem_bytes = rt_engine_.EstimateMemoryUsageForAABB(
        n_mbrs, config_.fast_build, config_.compact);
    buffer_.Init(mem_bytes * 1.2);
    auto gas_handle = BuildBVH(stream, mbrs_b, radius);
    stream.Sync();
    sw.stop();

    stats["BVHBuildTime"] = sw.ms();
    stats["BVHMemoryKB"] = mem_bytes / 1024;

    stats["HDLowerBound"] = hd_lb;
    stats["HDUpperBound"] = hd_ub;
    stats["InitRadius"] = radius;

    int iter = 0;
    uint32_t in_size = n_points_a;
#ifdef PROFILING
    hit_counters_.resize(n_points_a, 0);
    point_counters_.resize(n_points_a, 0);
    stats["Profiling"] = true;
#else
    stats["Profiling"] = false;
#endif
    in_queue_.Init(n_points_a);
    in_queue_.SetSequence(stream.cuda_stream(), in_size);
    term_queue_.Init(n_points_a);
    miss_queue_.Init(n_points_a);
    miss_queue_.Clear(stream.cuda_stream());

    while (in_size > 0) {
      iter++;
      auto& json_iter = stats["Iter" + std::to_string(iter)];

      term_queue_.Clear(stream.cuda_stream());

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
      params.mbrs_b = mbrs_b;
      params.prefix_sum = grid_.get_prefix_sum();
      params.point_b_ids = grid_.get_point_ids();
#ifdef PROFILING
      params.hit_counters = thrust::raw_pointer_cast(hit_counters_.data());
      params.point_counters = thrust::raw_pointer_cast(point_counters_.data());
#else
      params.hit_counters = nullptr;
      params.point_counters = nullptr;
#endif
      params.max_kcycles = limit_kcycles * block_size;
      params.prune = true;
      params.eb = true;

      rt_engine_.CopyLaunchParams(stream.cuda_stream(), params);
      rt_engine_.Render(stream.cuda_stream(), getRTModule(),
                        dim3{in_size, 1, 1});
      auto miss_size = miss_queue_.size(stream.cuda_stream());
      sw.stop();

      // Too many iterations, use EB to solve all points
      if (miss_size > 0 && iter >= miss_size) {
        term_queue_.Append(stream.cuda_stream(), miss_queue_);
        miss_queue_.Clear(stream.cuda_stream());

        VLOG(1) << "Transferring " << miss_size << " point to term queue";
        miss_size = 0;
      }

      auto term_size = term_queue_.size(stream.cuda_stream());

      json_iter["NumInputPoints"] = in_size;
      json_iter["NumOutputPoints"] = miss_size;
      json_iter["NumTermPoints"] = term_size;
      json_iter["CMax2"] = cmax2_.get(stream.cuda_stream());
      json_iter["RTTime"] = sw.ms();
      json_iter["Radius"] = radius;

      VLOG(1) << "Iter " << iter << " In " << in_size << " Miss " << miss_size
              << " Term " << term_size;

      // Need to run EB
      if (term_size > 0) {
        sw.start();
        ArrayView<uint32_t> eb_point_a_ids(term_queue_.data(), term_size);
        thrust::shuffle(thrust::cuda::par.on(stream.cuda_stream()),
                        eb_point_a_ids.begin(), eb_point_a_ids.end(), g_);
        CalculateHDEarlyBreak(stream, points_a, points_b_shuffled,
                              eb_point_a_ids);
        stream.Sync();
        sw.stop();
        json_iter["EBTime"] = sw.ms();
      } else {
        json_iter["ComparedPairs"] = 0;
        json_iter["EBTime"] = 0;
      }

      in_queue_.Clear(stream.cuda_stream());
      in_queue_.Swap(miss_queue_);

      uint32_t last_in_size = in_size;
      in_size = in_queue_.size(stream.cuda_stream());

      if (in_size > 0 && radius < hd_ub) {
        if (last_in_size == in_size) {
          radius *= 2;
        } else {
          radius += grid_.GetCellDigonalLength();
        }
        radius = std::min(radius, hd_ub);
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
    auto cmax2 = cmax2_.get(stream.cuda_stream());
    sw_total.stop();

    stats["Algorithm"] = "Hybrid";
    stats["Execution"] = "GPU";
    stats["ReportedTime"] = sw_total.ms();

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

  int CalculateHDEarlyBreak(
      const Stream& stream, const thrust::device_vector<point_t>& points_a,
      const thrust::device_vector<point_t>& points_b,
      ArrayView<uint32_t> v_point_ids_a = ArrayView<uint32_t>(),
      uint32_t* p_counters = nullptr, uint32_t* p_iter_counters = nullptr) {
    auto* p_cmax2 = cmax2_.data();
    ArrayView<point_t> v_points_a(points_a);
    ArrayView<point_t> v_points_b(points_b);

    uint32_t n_points_a = v_point_ids_a.size();

    if (n_points_a == 0) {
      n_points_a = points_a.size();
    }

    int2 launch_dims = LaunchKernel(stream, [=] __device__() {
      using BlockReduce = cub::BlockReduce<coord_t, MAX_BLOCK_SIZE>;
      __shared__ typename BlockReduce::TempStorage temp_storage;
      __shared__ bool early_break;
      __shared__ point_t point_a;

      for (auto i = blockIdx.x; i < n_points_a; i += gridDim.x) {
        auto size_b_roundup =
            div_round_up(v_points_b.size(), blockDim.x) * blockDim.x;

        auto begin_clk = clock();
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
        auto agg_pairs = BlockReduce(temp_storage).Reduce(n_pairs, cub::Sum());

        if (threadIdx.x == 0) {
          if (!early_break &&
              agg_cmin2 != std::numeric_limits<coord_t>::max()) {
            atomicMax(p_cmax2, agg_cmin2);
          }
          if (p_counters != nullptr) {
            p_counters[i] = clock() - begin_clk;
          }
          if (p_iter_counters != nullptr) {
            p_iter_counters[i] = n_iter;
          }
        }
      }
    });
    return launch_dims.y;  // block size
  }

 private:
  Config config_;
  thrust::default_random_engine g_;
  grid_t grid_;
  thrust::device_vector<OptixAabb> aabbs_;
  thrust::device_vector<uint32_t> hit_counters_, point_counters_;
  SharedValue<mbr_t> mbr_;
  Queue<uint32_t> in_queue_, term_queue_, miss_queue_;
  ReusableBuffer buffer_;
  details::RTEngine rt_engine_;
  SharedValue<COORD_T> cmax2_;
  Sampler sampler_;
  cudaDeviceProp dev_prop_;

  double get_sample_rate() const {
    auto sample_rate = config_.sample_rate;

    if (config_.auto_tune) {
      const auto& json_input = RunningStats::instance().Get("Input");
      FeaturesStatic<8> features(json_input, N_DIMS);
      auto feature_vals = features.Serialize();

      sample_rate = PredictSampleRate_3D(feature_vals.data());
      sample_rate = std::max(0.00001f, sample_rate);
    }
    return sample_rate;
  }

  uint32_t get_n_points_cell() const {
    auto n_points_cell = config_.n_points_cell;

    if (config_.auto_tune) {
      const auto& json_input = RunningStats::instance().Get("Input");
      FeaturesStatic<8> features(json_input, N_DIMS);
      auto feature_vals = features.Serialize();

      n_points_cell = PredictNumPointsPerCell_3D(feature_vals.data());
      n_points_cell = std::max(1, n_points_cell);
    }
    return n_points_cell;
  }

  details::ModuleIdentifier getRTModule() {
    details::ModuleIdentifier mod_nn = details::NUM_MODULE_IDENTIFIERS;

    if (typeid(COORD_T) == typeid(float)) {
      if (N_DIMS == 2) {
        mod_nn = details::MODULE_ID_FLOAT_NN_UNIFORM_GRID_2D;
      } else if (N_DIMS == 3) {
        mod_nn = details::MODULE_ID_FLOAT_NN_UNIFORM_GRID_3D;
      }
    } else if (typeid(COORD_T) == typeid(double)) {
      if (N_DIMS == 2) {
        mod_nn = details::MODULE_ID_DOUBLE_NN_UNIFORM_GRID_2D;
      } else if (N_DIMS == 3) {
        mod_nn = details::MODULE_ID_DOUBLE_NN_UNIFORM_GRID_3D;
      }
    }
    return mod_nn;
  }
};
}  // namespace hd

#endif  // HAUSDORFF_DISTANCE_HYBRID_H
