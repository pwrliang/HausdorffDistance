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
#include "loader.h"
#include "loaders/img_loader.h"
#include "move_points.h"
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
      // AutoTuneHausdorffDistanceImpl<float, 2>(config);
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
  itk::Size<N_DIMS> img_size_a, img_size_b;
  Stream stream;

  switch (config.input_type) {
  case InputType::kImage: {
    points_a = LoadImage<COORD_T, N_DIMS>(config.input_file1, img_size_a,
                                          config.limit);
    points_b = LoadImage<COORD_T, N_DIMS>(config.input_file2, img_size_b,
                                          config.limit);
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

  LOG(INFO) << "Points A: " << points_a.size()
            << " Points B: " << points_b.size();
  if (config.move_offset != 0) {
    MovePoints(points_a, points_b, config.move_offset);
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
  json_gpu["l2CacheSize"] = prop.l2CacheSize;
  json_gpu["multiProcessorCount"] = prop.multiProcessorCount;
  json_gpu["regsPerBlock"] = prop.regsPerBlock;
  json_gpu["maxThreadsPerBlock"] = prop.maxThreadsPerBlock;
  json_gpu["maxBlocksPerMultiProcessor"] = prop.maxBlocksPerMultiProcessor;
  json_gpu["regsPerMultiprocessor"] = prop.regsPerMultiprocessor;

  auto& json_input = stats.Log("Input");

  json_input["FileA"]["Path"] = config.input_file1;
  json_input["FileA"]["NumPoints"] = points_a.size();
  json_input["FileB"]["Path"] = config.input_file2;
  json_input["FileB"]["NumPoints"] = points_b.size();
  json_input["SerializationPrefix"] = config.serialize_folder;
  json_input["Limit"] = config.limit;
  json_input["NumDims"] = N_DIMS;
  json_input["Type"] = typeid(COORD_T) == typeid(float) ? "Float" : "Double";

  json_input["MoveOffset"] = config.move_offset;

  // Calculate MBR of points
  auto write_points_stats = [&](const std::string& key,
                                const thrust::device_vector<point_t>& points) {
    SharedValue<mbr_t> mbr;
    auto* p_mbr = mbr.data();

    UniformGrid<COORD_T, N_DIMS> stats_grid;

    mbr.set(stream.cuda_stream(), mbr_t());
    thrust::for_each(thrust::cuda::par.on(stream.cuda_stream()), points.begin(),
                     points.end(), [=] __device__(const point_t& p) mutable {
                       p_mbr->ExpandAtomic(p);
                     });
    auto h_mbr = mbr.get(stream.cuda_stream());

    auto grid_size = stats_grid.CalculateGridResolution(
        h_mbr, points.size(), config.stats_n_points_cell);

    stats_grid.Init(grid_size, h_mbr);
    stats_grid.Insert(stream, points);
    stats_grid.ComputeHistogram();

    auto& dataset_stats_json = json_input[key];

    dataset_stats_json["Grid"] = stats_grid.GetStats();
    auto json_mbr = nlohmann::json::array();

    for (int dim = 0; dim < N_DIMS; ++dim) {
      json_mbr.push_back(
          {{"Lower", h_mbr.lower(dim)}, {"Upper", h_mbr.upper(dim)}});
    }
    dataset_stats_json["MBR"] = json_mbr;
    dataset_stats_json["Density"] = points.size() / h_mbr.get_volume();
    return h_mbr;
  };

  thrust::device_vector<point_t> d_points_a = points_a, d_points_b = points_b;

  mbr_t merged_mbr = write_points_stats("FileA", d_points_a);
  merged_mbr.Expand(write_points_stats("FileB", d_points_b));
  json_input["Density"] =
      (points_a.size() + points_b.size()) / merged_mbr.get_volume();

  auto n_combinations =
      config.n_points_cell_list.size() * config.sample_rate_list.size() *
      config.max_hit_list.size() * config.radius_step_list.size() *
      config.sort_rays_list.size() * config.fast_build_bvh_list.size() *
      config.rebuild_bvh_list.size();
  uint32_t n_progress = 0;
  uint32_t n_skips = 0;
  double best_running_time = std::numeric_limits<double>::max();

  using hd_impl_t = HausdorffDistanceHybrid<COORD_T, N_DIMS>;
  typename hd_impl_t::Config hd_config;

  std::string ptx_root = config.exec_path + "/ptx";

  hd_config.ptx_root = ptx_root.c_str();

  auto hausdorff_distance = std::make_unique<hd_impl_t>(hd_config);

  Stopwatch sw_begin;
  sw_begin.start();
  std::ostringstream oss;
  for (size_t i = 0; i < config.max_hit_list.size(); ++i) {
    if (i != 0)
      oss << ',';
    oss << config.max_hit_list[i];
  }
  auto s_max_hit_list = oss.str();

  for (auto n_points_cell : config.n_points_cell_list) {
    for (auto sample_rate : config.sample_rate_list) {
      for (auto radius_step : config.radius_step_list) {
        for (auto sort_rays : config.sort_rays_list) {
          for (auto fast_build_bvh : config.fast_build_bvh_list) {
            for (auto rebuild_bvh : config.rebuild_bvh_list) {
              CHECK_GT(n_points_cell, 0) << "Avg points / cell cannot be zero";
              CHECK_GT(radius_step, 1);
              CHECK_LE(sample_rate, 1);

              VLOG(1) << "N_points_cell = " << n_points_cell
                      << ", sample_rate = " << sample_rate
                      << ", max_hit_list = " << s_max_hit_list
                      << ", radius = " << radius_step
                      << ", sort_rays = " << sort_rays
                      << ", fast_build_bvh = " << fast_build_bvh
                      << ", rebuild_bvh = " << rebuild_bvh;

              char path[PATH_MAX];
              sprintf(path,
                      "%s_n_points_cell_%u_sample_rate_%.6f_max_hit_list_%s_"
                      "radius_step_%.2f_sort_rays_%d_fast_build_bvh_%d_rebuild_"
                      "bvh_%d.json",
                      config.json_file.c_str(), n_points_cell, sample_rate,
                      s_max_hit_list.c_str(), radius_step, sort_rays,
                      fast_build_bvh, rebuild_bvh);
              bool file_exists = access(path, R_OK) == 0;

              if (!config.json_file.empty() && !config.overwrite &&
                  file_exists) {
                n_skips++;
                continue;
              }

              stats.Log("DateTime", get_current_datetime_string());

              auto& json_run = stats.Log("Running");

              json_run.clear();
              json_run["StatsNumPointsPerCell"] = config.stats_n_points_cell;
              json_run["Seed"] = config.seed;
              json_run["SortRays"] = sort_rays;
              json_run["FastBuildBVH"] = fast_build_bvh;
              json_run["RebuildBVH"] = rebuild_bvh;
              json_run["RadiusStep"] = radius_step;
              json_run["SampleRate"] = sample_rate;
              json_run["MaxHitList"] = s_max_hit_list;
              json_run["NumPointsPerCell"] = n_points_cell;

              COORD_T dist = -1;

              hd_config.sort_rays = sort_rays;
              hd_config.fast_build = fast_build_bvh;
              hd_config.rebuild_bvh = rebuild_bvh;
              hd_config.radius_step = radius_step;
              hd_config.sample_rate = sample_rate;
              hd_config.n_points_cell = n_points_cell;

              hausdorff_distance->UpdateConfig(hd_config);

              double running_time = 0;

              for (int i = 0; i < config.repeat; i++) {
                auto& json_repeat = json_run["Repeat" + std::to_string(i)];
                json_run["Variant"] = "Hybrid";
                json_run["Execution"] = "GPU";
                dist = hausdorff_distance->CalculateDistance(
                    stream, d_points_a, d_points_b, config.max_hit_list);
                json_repeat = hausdorff_distance->get_stats();
                auto total_time = json_repeat.at("TotalTime").get<double>();
                running_time += total_time;
              }
              best_running_time =
                  std::min(best_running_time, running_time / config.repeat);
              json_run["AvgTime"] = running_time / config.repeat;
              LOG(INFO) << std::fixed << std::setprecision(2)
                        << "Avg Running Time " << running_time / config.repeat
                        << " ms";

              stats.Log("HDResult", dist);

              if (config.check) {
                using hd_reference_impl =
                    HausdorffDistanceEarlyBreak<COORD_T, N_DIMS>;
                auto& json_check = stats.Log("Check");
                typename hd_reference_impl::Config hd_config;

                hd_config.n_threads = std::thread::hardware_concurrency();
                auto hd_reference = std::make_unique<
                    HausdorffDistanceEarlyBreak<COORD_T, N_DIMS>>(hd_config);
                auto answer_dist =
                    hd_reference->CalculateDistance(points_a, points_b);
                auto diff = answer_dist - dist;

                json_check["HDAnswer"] = answer_dist;
                if (dist != answer_dist) {
                  LOG(ERROR) << std::fixed << std::setprecision(8)
                             << "Wrong HausdorffDistance. Result: " << dist
                             << " Answer: " << answer_dist << " Diff: " << diff;
                } else {
                  LOG(INFO) << "HausdorffDistance is checked";
                }
                json_check["Diff"] = diff;
                json_check["Pass"] = dist == answer_dist;
              }

              if (!config.json_file.empty()) {
                if (!file_exists || file_exists && config.overwrite) {
                  stats.Dump(path);
                } else {
                  LOG(WARNING) << "Skip writting to JSON file " << path;
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
  }
}
}  // namespace hd
