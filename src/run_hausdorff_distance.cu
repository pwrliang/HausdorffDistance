#include <optix_function_table_definition.h>  // for g_optixFunctionTable

#include <algorithm>  // For std::shuffle
#include <chrono>
#include <cstdio>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <random>  // For random number generators
#include <sstream>

#include "hausdorff_distance_cpu.h"
#include "hausdorff_distance_gpu.h"
#include "hausdorff_distance_itk.h"
#include "hausdorff_distance_lbvh.h"
#include "hausdorff_distance_rt.h"
#include "img_loader.h"
#include "loader.h"
#include "models/features.h"
#include "models/tree_sample_rate.h"
#include "move_points.h"
#include "run_config.h"
#include "run_hausdorff_distance.cuh"
#include "running_stats.h"
#include "utils/stopwatch.h"
#include "utils/type_traits.h"

namespace hd {
template <typename COORD_T, int N_DIMS>
COORD_T RunHausdorffDistanceImpl(RunConfig config);

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

void RunHausdorffDistance(const RunConfig& config) {
  double dist = -1;

  if (config.is_double) {
    if (config.n_dims == 2) {
      // dist = RunHausdorffDistanceImpl<double, 2>(config);
    } else if (config.n_dims == 3) {
      // dist = RunHausdorffDistanceImpl<double, 3>(config);
    }
  } else {
    if (config.n_dims == 2) {
      dist = RunHausdorffDistanceImpl<float, 2>(config);
    } else if (config.n_dims == 3) {
      dist = RunHausdorffDistanceImpl<float, 3>(config);
    }
  }
  LOG(INFO) << "HausdorffDistance: distance is " << dist;
}

RunConfig PredicateBestConfig(double* features, const RunConfig& config) {
  RunConfig auto_tune_config = config;
  double sample_rate = PredicateSampleRate(features);

  LOG(INFO) << "User's Config, Auto-tune Config";
  LOG(INFO) << "Sample Rate: " << config.sample_rate << ", " << sample_rate;

  auto_tune_config.sample_rate = sample_rate;
  return auto_tune_config;
}

template <typename COORD_T, int N_DIMS>
COORD_T RunHausdorffDistanceImpl(RunConfig config) {
  using point_t = typename cuda_vec<COORD_T, N_DIMS>::type;
  std::vector<point_t> points_a, points_b;
  Stream stream;
  RunningStats& stats = RunningStats::instance();

  stats.Log("DateTime", get_current_datetime_string());
  auto& json_input = stats.Log("Input");

  switch (config.input_type) {
  case InputType::kImage: {
    points_a = LoadImage<COORD_T, N_DIMS>(config.input_file1, config.limit);
    points_b = LoadImage<COORD_T, N_DIMS>(config.input_file2, config.limit);
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

  json_input["FileA"]["Path"] = config.input_file1;
  json_input["FileA"]["NumPoints"] = points_a.size();
  json_input["FileB"]["Path"] = config.input_file2;
  json_input["FileB"]["NumPoints"] = points_b.size();
  json_input["SerializationPrefix"] = config.serialize_folder;
  json_input["Limit"] = config.limit;
  json_input["NumDims"] = N_DIMS;
  json_input["Type"] = typeid(COORD_T) == typeid(float) ? "Float" : "Double";

  if (config.move_offset != 0) {
    MovePoints(points_a, points_b, config.move_offset);
  }
  json_input["MoveOffset"] = config.move_offset;
  thrust::device_vector<point_t> d_points_a = points_a, d_points_b = points_b;

  // Calculate MBR of points
  auto write_points_stats = [&](const std::string& key,
                                const thrust::device_vector<point_t>& points) {
    using mbr_t = MbrTypeFromPoint<point_t>;

    SharedValue<mbr_t> mbr;
    auto* p_mbr = mbr.data();

    Grid<COORD_T, N_DIMS> stats_grid;

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
  };

  write_points_stats("FileA", d_points_a);
  write_points_stats("FileB", d_points_b);

  if (config.auto_tune) {
    CHECK(config.variant == Variant::kHybrid)
        << "You can only use auto-tune for the hybrid variant";
    Features<N_DIMS, 10> features(json_input);
    auto feature_vals = features.Serialize();
    config = PredicateBestConfig(feature_vals.data(), config);
  }

  auto& json_run = stats.Log("Running");

  json_run["StatsNumPointsPerCell"] = config.stats_n_points_cell;
  json_run["Seed"] = config.seed;
  json_run["SortRays"] = config.sort_rays;
  json_run["FastBuildBVH"] = config.fast_build_bvh;
  json_run["RebuildBVH"] = config.rebuild_bvh;
  json_run["RadiusStep"] = config.radius_step;
  json_run["SampleRate"] = config.sample_rate;
  json_run["MaxHit"] = config.max_hit;
  json_run["NumPointsPerCell"] = config.n_points_cell;

  COORD_T dist = -1;

  HausdorffDistanceRT<COORD_T, N_DIMS> hdist_rt;
  HausdorffDistanceLBVH<COORD_T, N_DIMS> hdist_lbvh;
  HausdorffDistanceRTConfig rt_config;
  std::string ptx_root = config.exec_path + "/ptx";

  rt_config.seed = config.seed;
  rt_config.ptx_root = ptx_root.c_str();
  rt_config.sort_rays = config.sort_rays;
  rt_config.fast_build = config.fast_build_bvh;
  rt_config.rebuild_bvh = config.rebuild_bvh;
  rt_config.radius_step = config.radius_step;
  rt_config.sample_rate = config.sample_rate;
  rt_config.max_reg_count = config.max_reg_count;
  rt_config.max_hit = config.max_hit;
  rt_config.n_points_cell = config.n_points_cell;

  hdist_rt.Init(rt_config);
  // hdist_lbvh.SetPointsTo(stream, points_b.begin(), points_b.end());

  LOG(INFO) << "Points A: " << points_a.size()
            << " Points B: " << points_b.size();
  Stopwatch sw;

  double running_time = 0;

  for (int i = 0; i < config.repeat; i++) {
    auto& json_repeat = json_run["Repeat" + std::to_string(i)];
    double loading_time_ms = 0;  // For ITK only

    sw.start();
    switch (config.variant) {
    case Variant::kEARLY_BREAK: {
      json_run["Variant"] = "EarlyBreak";
      switch (config.execution) {
      case Execution::kSerial:
        json_run["Execution"] = "Serial";
        dist = CalculateHausdorffDistance(points_a, points_b);
        break;
      case Execution::kParallel:
        json_run["Execution"] = "Parallel";
        dist = CalculateHausdorffDistanceParallel(points_a, points_b);
        break;
      case Execution::kGPU:
        json_run["Execution"] = "GPU";
        dist = CalculateHausdorffDistanceGPU<point_t>(
            stream, d_points_a, d_points_b, config.seed, json_repeat);
        break;
      }
      break;
    }
    case Variant::kZORDER: {
      json_run["Variant"] = "ZOrder";
      switch (config.execution) {
      case Execution::kSerial:
        json_run["Execution"] = "Serial";
        dist = CalculateHausdorffDistanceZOrder(points_a, points_b);
        break;
      case Execution::kGPU:
        json_run["Execution"] = "GPU";
        dist =
            CalculateHausdorffDistanceZorderGPU(stream, d_points_a, d_points_b);
        break;
      }
      break;
    }
    case Variant::kYUAN: {
      json_run["Variant"] = "Yuan";
      switch (config.execution) {
      case Execution::kSerial:
        json_run["Execution"] = "Serial";
        dist = CalculateHausdorffDistanceYuan(points_a, points_b);
        break;
      }
      break;
    }
    case Variant::kRT: {
      json_run["Variant"] = "RT";
      json_run["Execution"] = "GPU";
      if (config.n_points_cell > 0) {
        dist =
            hdist_rt.CalculateDistanceCompress(stream, d_points_a, d_points_b);
      } else {
        dist = hdist_rt.CalculateDistance(stream, d_points_a, d_points_b);
      }
      json_repeat = hdist_rt.GetStats();
      break;
    }
    case Variant::kHybrid: {
      json_run["Variant"] = "Hybrid";
      json_run["Execution"] = "GPU";
      CHECK_GT(config.n_points_cell, 0) << "Avg points / cell cannot be zero";
      CHECK_GT(config.radius_step, 1);
      CHECK_LE(config.sample_rate, 1);
      dist = hdist_rt.CalculateDistanceHybrid(stream, d_points_a, d_points_b);
      json_repeat = hdist_rt.GetStats();
      break;
    }
    case Variant::kBRANCH_BOUND: {
      // dist = hdist_lbvh.CalculateDistanceFrom(stream, points_a.begin(),
      // points_a.end());
      break;
    }
    case Variant::kITK: {
      json_run["Variant"] = "ITK";
      json_run["Execution"] = "Parallel";
      dist = CalculateHausdorffDistanceITK<N_DIMS>(config.input_file1.c_str(),
                                                   config.input_file2.c_str(),
                                                   loading_time_ms);
      json_repeat["LoadingTime"] = loading_time_ms;
      break;
    }
    }
    sw.stop();
    json_repeat["TotalTime"] = sw.ms() - loading_time_ms;
    running_time += sw.ms() - loading_time_ms;
  }

  json_run["AvgTime"] = running_time / config.repeat;
  LOG(INFO) << "Avg Running Time " << json_run["AvgTime"] << " ms";

  stats.Log("HDResult", dist);

  if (config.check) {
    auto& json_check = stats.Log("Check");

    auto answer_dist = CalculateHausdorffDistanceParallel(points_a, points_b);
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
    bool file_exists = access(config.json_file.c_str(), R_OK) == 0;

    if (!file_exists || file_exists && config.overwrite) {
      stats.Dump(config.json_file);
    }
  }
  return dist;
}
}  // namespace hd
