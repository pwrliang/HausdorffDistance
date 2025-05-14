#ifndef HAUSDORFF_DISTANCE_RAY_TRACING_H
#define HAUSDORFF_DISTANCE_RAY_TRACING_H
#include <glog/logging.h>
#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <thrust/shuffle.h>

#include <sstream>

#include "flags.h"
#include "geoms/distance.h"
#include "geoms/hd_bounds.h"
#include "geoms/mbr.h"
#include "hausdorff_distance.h"
#include "hd_impl/primitive_utils.h"
#include "index/uniform_grid.h"
#include "rt/launch_parameters.h"
#include "rt/reusable_buffer.h"
#include "rt/rt_engine.h"
#include "sampler.h"
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
class HausdorffDistanceRayTracing : public HausdorffDistance<COORD_T, N_DIMS> {
  using base_t = HausdorffDistance<COORD_T, N_DIMS>;
  using coord_t = COORD_T;
  using point_t = typename base_t::point_t;
  using mbr_t = Mbr<COORD_T, N_DIMS>;
  using grid_t = UniformGrid<COORD_T, N_DIMS>;

 public:
  struct Config {
    const char* ptx_root;
    int seed = 0;
    bool fast_build = false;
    bool compact = false;
    bool rebuild_bvh = false;
    float sample_rate = 0.001;
    int n_points_cell = 8;
    uint32_t max_samples = 100 * 1000;
    int max_reg_count = 0;
  };

  HausdorffDistanceRayTracing() = default;

  explicit HausdorffDistanceRayTracing(const Config& config) : config_(config) {
    auto rt_config = details::get_default_rt_config(config_.ptx_root);

    rt_config.max_reg_count = config_.max_reg_count;
    rt_engine_.Init(rt_config);
    sampler_ = Sampler(config_.seed);
    sampler_.Init(config_.max_samples);
    g_ = thrust::default_random_engine(config_.seed);
  }

  coord_t CalculateDistance(const Stream& stream,
                            thrust::device_vector<point_t>& points_a,
                            thrust::device_vector<point_t>& points_b) override {
    Stopwatch sw, sw_total;

    sw_total.start();
    uint64_t compared_points = 0;
    auto& stats = this->stats_;
    auto n_points_a = points_a.size();
    auto n_points_b = points_b.size();
    const auto mbr_a = CalculateMbr(stream, points_a.begin(), points_a.end());
    const auto mbr_b = CalculateMbr(stream, points_b.begin(), points_b.end());

    HdBounds<COORD_T, N_DIMS> hd_bounds(mbr_a);
    auto hd_lb = hd_bounds.GetLowerBound(mbr_b);
    auto hd_ub = hd_bounds.GetUpperBound(mbr_b);
    if (hd_ub == 0) {
      return 0;
    }
    cmax2_.set(stream.cuda_stream(), hd_lb * hd_lb);

    stats.clear();

    stats["Seed"] = config_.seed;
    stats["FastBuildBVH"] = config_.fast_build;
    stats["RebuildBVH"] = config_.rebuild_bvh;
    stats["SampleRate"] = config_.sample_rate;
    stats["NumPointsPerCell"] = config_.n_points_cell;

    COORD_T radius;

    // TODO: Build a Histogram over the sample-based EB
    // Find the median compared pairs
    // use the median as threshold

    // Sample points for a better initial HD
    {
      thrust::device_vector<point_t> points_b_shuffled = points_b;
      thrust::shuffle(thrust::cuda::par.on(stream.cuda_stream()),
                      points_b_shuffled.begin(), points_b_shuffled.end(), g_);
      sw.start();
      uint32_t n_samples = ceil(points_a.size() * config_.sample_rate);
      n_samples = std::min(n_samples, config_.max_samples);
      auto sampled_point_ids_a =
          sampler_.Sample(stream.cuda_stream(), points_a.size(), n_samples);
      CalculateHDEarlyBreak(stream, points_a, points_b_shuffled,
                            sampled_point_ids_a);
      auto sampled_hd2 = cmax2_.get(stream.cuda_stream());
      sw.stop();
      stats["NumSamples"] = n_samples;
      stats["SampleTime"] = sw.ms();
      stats["HD2AfterSampling"] = sampled_hd2;
      radius = sqrt(sampled_hd2);
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

    stats["HDLowerBound"] = hd_lb;
    stats["HDUpperBound"] = hd_ub;
    stats["InitRadius"] = radius;

    in_queue_.Init(n_points_a);
    out_queue_.Init(n_points_a);
    in_queue_.Clear(stream.cuda_stream());
    out_queue_.Clear(stream.cuda_stream());

    // Build BVH

    thrust::device_vector<mbr_t> mbrs_b;

    auto grid_size =
        grid_.CalculateGridResolution(mbr_b, n_points_b, config_.n_points_cell);
    grid_.Init(grid_size, mbr_b);
    grid_.Insert(stream, points_b);
    mbrs_b = grid_.GetTightCellMbrs(stream, points_b);

    stats["Grid"] = grid_.GetStats();

    sw.start();
    auto mem_bytes = rt_engine_.EstimateMemoryUsageForAABB(
        mbrs_b.size(), config_.fast_build, config_.compact);
    buffer_.Init(mem_bytes * 1.2);
    buffer_.Clear();
    auto gas_handle = BuildBVH(stream, mbrs_b, radius);
    stream.Sync();
    sw.stop();
    stats["BVHBuildTime"] = sw.ms();
    stats["BVHMemoryKB"] = mem_bytes / 1024;

    int iter = 0;
    uint32_t in_size = n_points_a;

#ifdef PROFILING
    hit_counters_.resize(n_points_a, 0);
    point_counters_.resize(n_points_a, 0);
    stats["Profiling"] = true;
#else
    stats["Profiling"] = false;
#endif

    in_queue_.SetSequence(stream.cuda_stream(), in_size);

    while (in_size > 0) {
      iter++;
      auto& json_iter = stats["Iter" + std::to_string(iter)];

      details::ModuleIdentifier mod_nn = getRTModule();

      sw.start();
      details::LaunchParamsNNUniformGrid<COORD_T, N_DIMS> params;

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
#ifdef PROFILING
      params.hit_counters = thrust::raw_pointer_cast(hit_counters_.data());
      params.point_counters = thrust::raw_pointer_cast(point_counters_.data());
#else
      params.hit_counters = nullptr;
      params.point_counters = nullptr;
#endif
      params.max_hit = std::numeric_limits<uint32_t>::max();

      rt_engine_.CopyLaunchParams(stream.cuda_stream(), params);
      rt_engine_.Render(stream.cuda_stream(), mod_nn, dim3{in_size, 1, 1});
      auto cmax2 = cmax2_.get(stream.cuda_stream());
      sw.stop();

      auto cmax = sqrt(cmax2);

      VLOG(1) << "Iter: " << iter << " radius: " << radius << std::fixed
              << std::setprecision(8) << " cmax2: " << cmax2
              << " cmax: " << cmax << " in_size: " << in_size
              << " out_size: " << out_queue_.size(stream.cuda_stream())
              << " Time: " << sw.ms() << " ms";

      json_iter["NumInputPoints"] = in_size;
      in_queue_.Clear(stream.cuda_stream());
      in_queue_.Swap(out_queue_);
      in_size = in_queue_.size(stream.cuda_stream());
      json_iter["NumOutputPoints"] = in_size;
      json_iter["CMax2"] = cmax2;
      json_iter["RTTime"] = sw.ms();
#ifdef PROFILING
      json_iter["AccumulatedHits"] = thrust::reduce(
          thrust::cuda::par.on(stream.cuda_stream()), hit_counters_.begin(),
          hit_counters_.end(), 0ul, thrust::plus<uint64_t>());
      json_iter["AccumulatedComparedPoints"] = thrust::reduce(
          thrust::cuda::par.on(stream.cuda_stream()), point_counters_.begin(),
          point_counters_.end(), 0ul, thrust::plus<uint64_t>());
#endif
      json_iter["Radius"] = radius;
      json_iter["CoveredCells"] = radius / grid_.GetCellDigonalLength();

      if (in_size > 0 && radius < hd_ub) {
        radius += grid_.GetCellDigonalLength();
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

    stats["Algorithm"] = "Ray Tracing";
    stats["Execution"] = "GPU";
    stats["ComparedPairs"] = compared_points;
    stats["ReportedTime"] = sw_total.ms();

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

  void CalculateHDEarlyBreak(
      const Stream& stream, const thrust::device_vector<point_t>& points_a,
      const thrust::device_vector<point_t>& points_b,
      ArrayView<uint32_t> v_point_ids_a = ArrayView<uint32_t>()) {
    auto* p_cmax2 = cmax2_.data();
    ArrayView<point_t> v_points_a(points_a);
    ArrayView<point_t> v_points_b(points_b);

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
        }
      }
    });
  }

  const thrust::device_vector<uint32_t>& get_hit_counters() const {
    return hit_counters_;
  }

  const thrust::device_vector<uint32_t>& get_point_counters() const {
    return point_counters_;
  }

 private:
  Config config_;
  thrust::default_random_engine g_;
  grid_t grid_;
  thrust::device_vector<OptixAabb> aabbs_;
  thrust::device_vector<uint32_t> hit_counters_, point_counters_;
  SharedValue<mbr_t> mbr_;
  Queue<uint32_t> in_queue_, out_queue_;
  ReusableBuffer buffer_;
  details::RTEngine rt_engine_;
  SharedValue<COORD_T> cmax2_;
  Sampler sampler_;

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

#endif  // HAUSDORFF_DISTANCE_RAY_TRACING_H
