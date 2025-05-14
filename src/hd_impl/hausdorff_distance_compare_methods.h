#ifndef HAUSDORFF_DISTANCE_COMPARE_METHODS_H
#define HAUSDORFF_DISTANCE_COMPARE_METHODS_H
#include <glog/logging.h>

#include <vector>

#ifndef PROFILING
#define PROFILING
#endif

#include "hd_impl/hausdorff_distance_early_break.h"
#include "hd_impl/hausdorff_distance_ray_tracing.h"

#ifdef PROFILING
#undef PROFILING
#endif

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
    float sample_rate = 0.001;
    int n_points_cell = 8;
    int max_samples = 100 * 1000;
    int max_reg_count = 0;
  };

  HausdorffDistanceCompareMethods() = default;
  explicit HausdorffDistanceCompareMethods(const Config& config) {
    typename rt_impl_t::Config rt_config;
    typename eb_impl_t::Config eb_config;

    rt_config.ptx_root = config.ptx_root;
    rt_config.seed = config.seed;
    rt_config.fast_build = config.fast_build;
    rt_config.compact = config.compact;
    rt_config.rebuild_bvh = config.rebuild_bvh;
    rt_config.sample_rate = config.sample_rate;
    rt_config.n_points_cell = config.n_points_cell;
    rt_config.max_samples = config.max_samples;
    rt_config.max_reg_count = config.max_reg_count;

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
    sw.start();
    auto hd_eb = early_break_->CalculateDistance(stream, points_a, points_b);
    sw.stop();
    LOG(INFO) << "EB Time " << sw.ms();

    // CHECK_EQ(hd_rt, hd_eb);

    if (hd_rt != hd_eb) {
      printf("Wrong answer!!!!! rt %f vs eb %f\n", hd_rt, hd_eb);
    }

    thrust::host_vector<uint32_t> rt_point_counters =
        ray_tracing_->get_point_counters();
    thrust::host_vector<uint32_t> rt_hit_counters =
        ray_tracing_->get_hit_counters();
    thrust::host_vector<uint32_t> eb_point_counters =
        early_break_->get_point_counters();

    size_t np_rt_less_work = 0;
    size_t eb_total_paris = 0, rt_total_pairs = 0;
    size_t rt_total_hits = 0;

    for (size_t i = 0; i < rt_point_counters.size(); i++) {
      rt_total_pairs += rt_point_counters[i];
      eb_total_paris += eb_point_counters[i];
      rt_total_hits += rt_hit_counters[i];
      if (rt_point_counters[i] < eb_point_counters[i]) {
        np_rt_less_work++;
      }
    }

    PrintHistogram(rt_hit_counters, 2);

    auto rt_gini = gini_index_thrust(stream, rt_point_counters);
    auto rt_git_gini = gini_index_thrust(stream, rt_hit_counters);
    auto eb_gini = gini_index_thrust(stream, eb_point_counters);

    LOG(INFO) << "RT Fast Ratio " << (float) np_rt_less_work / points_a.size();
    LOG(INFO) << "RT Total Pairs " << rt_total_pairs;
    LOG(INFO) << "RT Total Hits " << rt_total_hits;
    if (rt_total_hits > rt_total_pairs) {
      printf("More Hits than Pairs!!!!!\n");
    }
    LOG(INFO) << "RT Gini " << rt_gini;
    LOG(INFO) << "RT Hit Gini " << rt_git_gini;

    LOG(INFO) << "EB Total Pairs " << eb_total_paris;
    LOG(INFO) << "EB Gini " << eb_gini;

    stats["Early Break"] = early_break_->get_stats();
    stats["Ray Tracing"] = ray_tracing_->get_stats();

    stats["Algorithm"] = "Compare Methods";
    stats["Execution"] = "GPU";
    stats["ReportedTime"] = sw.ms();
    return hd_eb;
  }

 private:
  std::unique_ptr<rt_impl_t> ray_tracing_;
  std::unique_ptr<eb_impl_t> early_break_;

  void PrintHistogram(const thrust::host_vector<uint32_t>& vec,
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

    hdr_percentiles_print(histogram, stdout, ticks_per_half_distance, 1.0,
                          CLASSIC);
    hdr_close(histogram);
  }
};
}  // namespace hd

#endif  // HAUSDORFF_DISTANCE_COMPARE_METHODS_H
