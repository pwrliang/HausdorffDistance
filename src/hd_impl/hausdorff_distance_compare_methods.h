#ifndef HAUSDORFF_DISTANCE_COMPARE_METHODS_H
#define HAUSDORFF_DISTANCE_COMPARE_METHODS_H
#include <glog/logging.h>

#include <vector>

#ifndef PROFILING
#define PROFILING
#endif

#include "hd_impl/hausdorff_distance_early_break.h"
#include "hd_impl/hausdorff_distance_ray_tracing.h"

namespace hd {
template <typename COORD_T, int N_DIMS>
class HausdorffDistanceCompareMethods
    : public HausdorffDistance<COORD_T, N_DIMS> {
  using rt_impl_t = HausdorffDistanceRayTracing<COORD_T, N_DIMS>;
  using eb_impl_t = HausdorffDistanceEarlyBreak<COORD_T, N_DIMS>;
  using point_t = typename HausdorffDistance<COORD_T, N_DIMS>::point_t;

 public:
  struct Config {
    const char* ptx_root;
    int seed = 0;
    bool fast_build = false;
    bool compact = false;
    bool rebuild_bvh = false;
    int n_points_cell = 8;
    int max_reg_count = 0;
    uint32_t max_hit = std::numeric_limits<uint32_t>::max();
    bool prune = true;
    bool eb = true;
  };

  HausdorffDistanceCompareMethods() = default;
  explicit HausdorffDistanceCompareMethods(const Config& config) {
    typename rt_impl_t::Config rt_config;
    typename eb_impl_t::Config eb_config;

    rt_config.ptx_root = config.ptx_root;
    rt_config.fast_build = config.fast_build;
    rt_config.compact = config.compact;
    rt_config.rebuild_bvh = config.rebuild_bvh;
    rt_config.n_points_cell = config.n_points_cell;
    rt_config.max_reg_count = config.max_reg_count;
    rt_config.max_hit = config.max_hit;
    rt_config.prune = config.prune;
    rt_config.eb = config.eb;
    eb_config.seed = config.seed;

    ray_tracing_ = std::make_unique<rt_impl_t>(rt_config);
    early_break_ = std::make_unique<eb_impl_t>(eb_config);
  }

  COORD_T CalculateDistance(const Stream& stream,
                            thrust::device_vector<point_t>& points_a,
                            thrust::device_vector<point_t>& points_b) override {
    auto& stats = this->stats_;
    Stopwatch sw;
    sw.start();
    auto hd_rt = ray_tracing_->CalculateDistance(stream, points_a, points_b);
    sw.stop();
    LOG(INFO) << "RT Time " << sw.ms();
    // sw.start();
    // auto hd_eb = early_break_->CalculateDistance(stream, points_a, points_b);
    // sw.stop();
    // LOG(INFO) << "EB Time " << sw.ms();

    // CHECK_EQ(hd_rt, hd_eb);

    // thrust::host_vector<uint32_t> rt_point_counters =
    //     ray_tracing_->get_point_counters();
    thrust::host_vector<uint32_t> rt_hit_counters =
    ray_tracing_->get_hit_counters();
    // thrust::host_vector<uint32_t> eb_point_counters =
        // early_break_->get_point_counters();



    // stats["Early Break"] = early_break_->get_stats();
    auto json_stats = ray_tracing_->get_stats();
    json_stats["HitHisto"] = PrintHistogram(rt_hit_counters, 2);

    stats["Ray Tracing"] = json_stats;
    stats["Algorithm"] = "Compare Methods";
    stats["Execution"] = "GPU";
    stats["ReportedTime"] = sw.ms();
    return hd_rt;
  }

 private:
  std::unique_ptr<rt_impl_t> ray_tracing_;
  std::unique_ptr<eb_impl_t> early_break_;

  nlohmann::json PrintHistogram(const thrust::host_vector<uint32_t>& vec,
                      int ticks_per_half_distance) {
    hdr_histogram* histogram;
    // Initialise the histogram
    hdr_init(1,                     // Minimum value
             (int64_t) vec.size(),  // Maximum value
             3,                     // Number of significant figures
             &histogram);           // Pointer to initialise

    for (auto val : vec) {
      if (val > 0)
        hdr_record_value(histogram,  // Histogram to record to
                         val);       // Value to record
    }

    auto json = DumpHistogram(histogram);

    // hdr_percentiles_print(histogram, stdout, ticks_per_half_distance, 1.0,
                          // CLASSIC);
    hdr_close(histogram);
    return json;
  }
};
}  // namespace hd

#ifdef PROFILING
#undef PROFILING
#endif

#endif  // HAUSDORFF_DISTANCE_COMPARE_METHODS_H
