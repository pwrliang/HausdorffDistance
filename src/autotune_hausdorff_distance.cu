#include <glog/logging.h>

#include <algorithm>  // For std::shuffle
#include <chrono>
#include <cstdio>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <random>  // For random number generators
#include <sstream>

#include "hausdorff_distance.h"
#include "hd_impl/hausdorff_distance_early_break.h"
#include "hd_impl/hausdorff_distance_hybrid.h"
#include "loaders/img_loader.h"
#include "loaders/loader.h"
#include "loaders/ply_loader.h"
#include "loaders/translate_points.h"
#include "run_config.h"
#include "running_stats.h"
#include "utils/stopwatch.h"
#include "utils/type_traits.h"

namespace hd {
template <typename COORD_T, int N_DIMS>
void AutoTuneHausdorffDistanceImpl(const RunConfig& config);

inline std::string get_current_datetime_string() {
  auto now = std::chrono::system_clock::now();
  std::time_t time_now = std::chrono::system_clock::to_time_t(now);

  std::tm local_tm;
#if defined(_MSC_VER)
  localtime_s(&local_tm, &time_now);  // MSVC
#else
  localtime_r(&time_now, &local_tm);  // POSIX
#endif

  std::ostringstream oss;
  oss << std::put_time(&local_tm, "%Y-%m-%d %H:%M:%S");
  return oss.str();
}

void AutoTuneHausdorffDistance(const RunConfig& config) {
  if (config.is_double) {
    if (config.n_dims == 2) {
      // dist = RunHausdorffDistanceImpl<double, 2>(config);
    } else if (config.n_dims == 3) {
      // dist = RunHausdorffDistanceImpl<double, 3>(config);
    }
  } else {
    if (config.n_dims == 2) {
      AutoTuneHausdorffDistanceImpl<float, 2>(config);
    } else if (config.n_dims == 3) {
      AutoTuneHausdorffDistanceImpl<float, 3>(config);
    }
  }
}

template <typename COORD_T, int N_DIMS>
void AutoTuneHausdorffDistanceImpl(const RunConfig& config) {
  using point_t = typename cuda_vec<COORD_T, N_DIMS>::type;
  using mbr_t = MbrTypeFromPoint<point_t>;
  std::vector<point_t> points_a, points_b;
  Stream stream;

  switch (config.input_type) {
  case InputType::kImage: {
    itk::Size<N_DIMS> img_size_a, img_size_b;
    points_a = LoadImage<COORD_T, N_DIMS>(config.input_file1, img_size_a,
                                          config.limit);
    points_b = LoadImage<COORD_T, N_DIMS>(config.input_file2, img_size_b,
                                          config.limit);
    break;
  }
  case InputType::kPLY: {
    points_a = LoadPLY<COORD_T, N_DIMS>(config.input_file1, config.limit);
    points_b = LoadPLY<COORD_T, N_DIMS>(config.input_file2, config.limit);
    break;
  }
  default: {
    points_a =
        LoadPoints<COORD_T, N_DIMS>(config.input_file1, config.serialize_folder,
                                    config.input_type, config.limit);
    points_b =
        LoadPoints<COORD_T, N_DIMS>(config.input_file2, config.serialize_folder,
                                    config.input_type, config.limit);
    break;
  }
  }
  CHECK_GT(points_a.size(), 0) << config.input_file1;
  CHECK_GT(points_b.size(), 0) << config.input_file2;

  nlohmann::json json_file1;
  nlohmann::json json_file2;

  json_file1["Path"] = config.input_file1;
  json_file1["NumPoints"] = points_a.size();
  json_file2["Path"] = config.input_file2;
  json_file2["NumPoints"] = points_b.size();

  if (config.normalize) {
    NormalizePoints(points_a);
    NormalizePoints(points_b);
  }
  if (config.translate != 0) {
    // translate x
    TranslatePoints(points_b, 0, config.translate);
  }

  RunningStats& stats = RunningStats::instance();

  auto& json_gpu = stats.Log("GPU");
  cudaDeviceProp prop;
  int device;
  cudaGetDevice(&device);  // Get current device ID
  cudaGetDeviceProperties(&prop,
                          device);  // Get properties of the current device
  json_gpu["Device"] = device;
  json_gpu["name"] = prop.name;

  // Calculate MBR of points
  auto write_points_stats = [&](nlohmann::json& json_file,
                                const thrust::device_vector<point_t>& points) {
    SharedValue<mbr_t> mbr;
    auto* p_mbr = mbr.data();

    UniformGrid<COORD_T, N_DIMS> stats_grid;

    mbr.set(stream.cuda_stream(), mbr_t());
    thrust::for_each(thrust::cuda::par.on(stream.cuda_stream()), points.begin(),
                     points.end(), [=] __device__(const point_t& p) mutable {
                       p_mbr->ExpandAtomic(p);
                     });
    auto h_mbr = mbr.get(stream.cuda_stream()).ToNonemptyMBR();

    auto grid_size = stats_grid.CalculateGridResolution(
        h_mbr, points.size(), config.stats_n_points_cell);

    stats_grid.Init(grid_size, h_mbr);
    stats_grid.Insert(stream, points);
    stats_grid.ComputeHistogram();

    json_file["Grid"] = stats_grid.GetStats();
    json_file["StatsGridNumPointsPerCell"] = config.stats_n_points_cell;

    auto json_mbr = nlohmann::json::array();

    for (int dim = 0; dim < N_DIMS; ++dim) {
      json_mbr.push_back(
          {{"Lower", h_mbr.lower(dim)}, {"Upper", h_mbr.upper(dim)}});
    }
    json_file["MBR"] = json_mbr;
    json_file["Density"] = points.size() / h_mbr.get_volume();
    return h_mbr;
  };

  thrust::device_vector<point_t> d_points_a = points_a, d_points_b = points_b;

  mbr_t merged_mbr = write_points_stats(json_file1, d_points_a);
  merged_mbr.Expand(write_points_stats(json_file2, d_points_b));

  auto& json_input = stats.Log("Input");

  {
    pinned_vector<point_t> all_points = d_points_a;
    all_points.insert(all_points.end(), d_points_b.begin(), d_points_b.end());
    write_points_stats(json_input, all_points);
  }

  json_input["Files"] = {json_file1, json_file2};
  json_input["Normalize"] = config.normalize;
  json_input["Translate"] = config.translate;
  json_input["SerializationPrefix"] = config.serialize_folder;
  json_input["Limit"] = config.limit;
  json_input["NumDims"] = N_DIMS;
  json_input["Type"] = typeid(COORD_T) == typeid(float) ? "Float" : "Double";
  json_input["Density"] =
      (points_a.size() + points_b.size()) / merged_mbr.get_volume();

  using hd_impl_t = HausdorffDistanceHybrid<COORD_T, N_DIMS>;
  typename hd_impl_t::Config hd_config;

  std::string ptx_root = config.exec_path + "/ptx";

  hd_config.ptx_root = ptx_root.c_str();

  auto hausdorff_distance = std::make_unique<hd_impl_t>(hd_config);

  Stopwatch sw_begin;
  sw_begin.start();

  auto eb_only_threshold_list = config.eb_only_threshold_list;

  // too many points, eb is not used
  if (points_b.size() >= hd_config.sample_threshold) {
    eb_only_threshold_list.clear();
    eb_only_threshold_list.push_back(0);
  }

  double best_running_time = std::numeric_limits<double>::max();
  uint32_t n_progress = 0;
  uint32_t n_skips = 0;
  auto n_combinations =
      config.sample_rate_list.size() * eb_only_threshold_list.size() *
      config.max_hit_list.size() * config.n_points_cell_list.size();

  for (auto sample_rate : config.sample_rate_list) {
    for (auto eb_only_threshold : eb_only_threshold_list) {
      for (auto max_hit : config.max_hit_list) {
        for (auto n_points_cell : config.n_points_cell_list) {
          CHECK_GT(n_points_cell, 0) << "Avg points / cell cannot be zero";
          CHECK_LE(sample_rate, 1);

          VLOG(1) << "Sample rate: " << sample_rate
                  << " EB Only Threshold: " << eb_only_threshold
                  << " Max Hit: " << max_hit << " #Points/Cell "
                  << n_points_cell;

          char path[PATH_MAX];
          sprintf(path,
                  "%s_sample_rate_%.6f_eb_only_threshold_%u_max_hit_%u_n_"
                  "points_cell_%u.json",
                  config.json_file.c_str(), sample_rate, eb_only_threshold,
                  max_hit, n_points_cell);
          bool file_exists = access(path, R_OK) == 0;

          if (!config.json_file.empty() && !config.overwrite && file_exists) {
            LOG(INFO) << "Skip " << path;
            n_skips++;
            continue;
          }

          stats.Log("DateTime", get_current_datetime_string());

          auto& json_run = stats.Log("Running");

          json_run.clear();
          json_run["Seed"] = config.seed;
          json_run["SampleRate"] = sample_rate;
          json_run["EBOnlyThreshold"] = eb_only_threshold;
          json_run["MaxHit"] = max_hit;
          json_run["NumPointsPerCell"] = n_points_cell;

          COORD_T dist = -1;

          hd_config.sample_rate = sample_rate;
          hd_config.eb_only_threshold = eb_only_threshold;
          hd_config.max_hit = max_hit;
          hd_config.n_points_cell = n_points_cell;

          hausdorff_distance->UpdateConfig(hd_config);

          double running_time = 0;
          auto& json_repeats = json_run["Repeats"];

          for (int i = 0; i < config.repeat; i++) {
            json_repeats.push_back(nlohmann::json());
            auto& json_repeat = json_repeats.back();

            json_repeat["Repeat"] = i + 1;
            dist = hausdorff_distance->CalculateDistance(stream, d_points_a,
                                                         d_points_b);
            json_repeat = hausdorff_distance->get_stats();
            running_time += json_repeat.at("ReportedTime").get<double>();
          }
          best_running_time =
              std::min(best_running_time, running_time / config.repeat);
          json_run["AvgTime"] = running_time / config.repeat;
          LOG(INFO) << std::fixed << std::setprecision(2) << "Avg Running Time "
                    << running_time / config.repeat << " ms";

          stats.Log("HDResult", dist);

          if (!config.json_file.empty()) {
            if (!file_exists || file_exists && config.overwrite) {
              stats.Dump(path);
            } else {
              LOG(WARNING) << "Skip writing to JSON file " << path;
            }
          }

          n_progress++;
          sw_begin.stop();

          LOG(INFO) << "Progress " << std::fixed << std::setprecision(2)
                    << (float) (n_progress + n_skips) / n_combinations * 100
                    << " % Remaining Time "
                    << sw_begin.ms() / n_progress *
                           (n_combinations - n_progress - n_skips) / 1000
                    << " s Best Performance " << best_running_time << " ms";
        }
      }
    }
  }
}
}  // namespace hd
