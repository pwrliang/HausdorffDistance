#include <optix_function_table_definition.h>  // for g_optixFunctionTable

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>

#include "hd_impl/hausdorff_distance_branch_n_bound.h"
#include "hd_impl/hausdorff_distance_compare_methods.h"
#include "hd_impl/hausdorff_distance_early_break.h"
#include "hd_impl/hausdorff_distance_hybrid.h"
#include "hd_impl/hausdorff_distance_itk.h"
#include "hd_impl/hausdorff_distance_nearest_neighbor_search.h"
#include "hd_impl/hausdorff_distance_ray_tracing.h"
#include "hd_impl/hausdorff_distance_rt_hdist.h"
#include "loaders/img_loader.h"
#include "loaders/loader.h"
#include "loaders/ply_loader.h"
#include "loaders/translate_points.h"
#include "models/features.h"
#include "models/tree_numpointspercell_3d.h"
#include "models/tree_samplerate_3d.h"
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
  double sample_rate;
  double n_points_cell;

  sample_rate = PredictSampleRate_3D(features);
  n_points_cell = PredictNumPointsPerCell_3D(features);

  if (sample_rate <= 0) {
    sample_rate = 0.0001;
  }

  LOG(INFO) << "User's Config, Auto-tune Config";
  LOG(INFO) << "Sample Rate: " << config.sample_rate << ", " << sample_rate;
  LOG(INFO) << "Points/Cell: " << config.n_points_cell << ", " << n_points_cell;

  CHECK(sample_rate > 0 && sample_rate <= 1);
  CHECK(n_points_cell > 1);

  auto_tune_config.sample_rate = sample_rate;
  auto_tune_config.n_points_cell = n_points_cell;
  return auto_tune_config;
}

template <typename COORD_T, int N_DIMS>
COORD_T RunHausdorffDistanceImpl(RunConfig config) {
  using point_t = typename cuda_vec<COORD_T, N_DIMS>::type;
  using mbr_t = MbrTypeFromPoint<point_t>;
  std::vector<point_t> points_a, points_b;
  Stream stream;
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

  stats.Log("DateTime", get_current_datetime_string());
  auto& json_input = stats.Log("Input");
  itk::Size<N_DIMS> img_size_a, img_size_b;

  switch (config.input_type) {
  case InputType::kImage: {
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

  json_input["FileA"]["Path"] = config.input_file1;
  json_input["FileA"]["NumPoints"] = points_a.size();
  json_input["FileB"]["Path"] = config.input_file2;
  json_input["FileB"]["NumPoints"] = points_b.size();
  json_input["SerializationPrefix"] = config.serialize_folder;
  json_input["Limit"] = config.limit;
  json_input["NumDims"] = N_DIMS;
  json_input["Type"] = typeid(COORD_T) == typeid(float) ? "Float" : "Double";

#if 1
  if (config.move_to_origin || config.normalize) {
    MoveToOrigin(points_a);
    MoveToOrigin(points_b);
    if (config.normalize) {
      NormalizePoints(points_a);
      NormalizePoints(points_b);
    }
  }
  if (config.translate != 0) {
    // translate x
    TranslatePoints(points_b, 0, config.translate);
  }
  json_input["MoveToOrigin"] = config.move_to_origin;
  json_input["Normalize"] = config.normalize;
  json_input["Translate"] = config.translate;
  thrust::device_vector<point_t> d_points_a = points_a, d_points_b = points_b;

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
    dataset_stats_json["StatsGridNumPointsPerCell"] =
        config.stats_n_points_cell;

    auto json_mbr = nlohmann::json::array();

    for (int dim = 0; dim < N_DIMS; ++dim) {
      json_mbr.push_back(
          {{"Lower", h_mbr.lower(dim)}, {"Upper", h_mbr.upper(dim)}});
    }
    dataset_stats_json["MBR"] = json_mbr;
    dataset_stats_json["Density"] = points.size() / h_mbr.get_volume();
    return h_mbr;
  };

  mbr_t merged_mbr = write_points_stats("FileA", d_points_a);
  merged_mbr.Expand(write_points_stats("FileB", d_points_b));
  json_input["Density"] =
      (points_a.size() + points_b.size()) / merged_mbr.get_volume();

  if (config.auto_tune) {
    CHECK(config.variant == Variant::kHybrid)
        << "You can only use auto-tune for the hybrid variant";
    FeaturesStatic<N_DIMS, 8> features(json_input);
    auto feature_vals = features.Serialize();
    config = PredicateBestConfig(feature_vals.data(), config);
  }

  auto& json_run = stats.Log("Running");

  json_run["AutoTune"] = config.auto_tune;
  json_run["Seed"] = config.seed;
  json_run["FastBuildBVH"] = config.fast_build_bvh;
  json_run["RebuildBVH"] = config.rebuild_bvh;
  json_run["SampleRate"] = config.sample_rate;
  json_run["NumPointsPerCell"] = config.n_points_cell;

  COORD_T dist = -1;

  std::unique_ptr<HausdorffDistance<COORD_T, N_DIMS>> hausdorff_distance;

  if (config.parallelism <= 0) {
    config.parallelism = std::thread::hardware_concurrency();
  }

  switch (config.variant) {
  case Variant::kCompareMethods: {
    using hd_impl_t = HausdorffDistanceCompareMethods<COORD_T, N_DIMS>;
    typename hd_impl_t::Config hd_config;
    std::string ptx_root = config.exec_path + "/ptx";
    hd_config.seed = config.seed;
    hd_config.ptx_root = ptx_root.c_str();
    hd_config.fast_build = config.fast_build_bvh;
    hd_config.rebuild_bvh = config.rebuild_bvh;
    hd_config.sample_rate = config.sample_rate;
    hd_config.n_points_cell = config.n_points_cell;

    hausdorff_distance = std::make_unique<hd_impl_t>(hd_config);
    break;
  }
  case Variant::kEarlyBreak: {
    using hd_impl_t = HausdorffDistanceEarlyBreak<COORD_T, N_DIMS>;
    typename hd_impl_t::Config hd_config;

    hd_config.seed = config.seed;
    hd_config.n_threads = config.parallelism;
    hausdorff_distance =
        std::make_unique<HausdorffDistanceEarlyBreak<COORD_T, N_DIMS>>(
            hd_config);
    break;
  }
  case Variant::kHybrid: {
    using hd_impl_t = HausdorffDistanceHybrid<COORD_T, N_DIMS>;
    typename hd_impl_t::Config hd_config;
    std::string ptx_root = config.exec_path + "/ptx";
    hd_config.seed = config.seed;
    hd_config.ptx_root = ptx_root.c_str();
    hd_config.auto_tune = config.auto_tune;
    hd_config.fast_build = config.fast_build_bvh;
    hd_config.rebuild_bvh = config.rebuild_bvh;
    hd_config.sample_rate = config.sample_rate;
    hd_config.n_points_cell = config.n_points_cell;
    hd_config.max_hit_ratio = config.max_hit_ratio;

    hausdorff_distance = std::make_unique<hd_impl_t>(hd_config);
    break;
  }
  case Variant::kRT: {
    using hd_impl_t = HausdorffDistanceRayTracing<COORD_T, N_DIMS>;
    typename hd_impl_t::Config hd_config;
    std::string ptx_root = config.exec_path + "/ptx";
    hd_config.seed = config.seed;
    hd_config.ptx_root = ptx_root.c_str();
    hd_config.fast_build = config.fast_build_bvh;
    hd_config.rebuild_bvh = config.rebuild_bvh;
    hd_config.sample_rate = config.sample_rate;
    hd_config.n_points_cell = config.n_points_cell;

    hausdorff_distance = std::make_unique<hd_impl_t>(hd_config);
    break;
  }
#if 0
  case Variant::kBranchAndBound: {
    using hd_impl_t = HausdorffDistanceBranchNBound<COORD_T, N_DIMS>;

    hausdorff_distance = std::make_unique<hd_impl_t>();
    break;
  }
  case Variant::kNearestNeighborSearch: {
    using hd_impl_t = HausdorffDistanceNearestNeighborSearch<COORD_T, N_DIMS>;
    typename hd_impl_t::Config hd_config;

    hd_config.n_threads = config.parallelism;
    hausdorff_distance = std::make_unique<hd_impl_t>(hd_config);
    break;
  }
  case Variant::kRT_HDIST: {
    using hd_impl_t = HausdorffDistanceRTHDist<COORD_T, N_DIMS>;
    typename hd_impl_t::Config hd_config;
    std::string ptx_root = config.exec_path + "/ptx";
    hd_config.ptx_root = ptx_root.c_str();
    hd_config.fast_build = config.fast_build_bvh;
    hd_config.rebuild_bvh = config.rebuild_bvh;
    hd_config.bit_count = config.bit_count;

    hausdorff_distance = std::make_unique<hd_impl_t>(hd_config);
    break;
  }

  case Variant::kITK: {
    using hd_impl_t = HausdorffDistanceITK<COORD_T, N_DIMS>;
    typename hd_impl_t::Config hd_config;

    hd_config.size_a = img_size_a;
    hd_config.size_b = img_size_b;
    hd_config.n_threads = config.parallelism;

    hausdorff_distance = std::make_unique<hd_impl_t>(hd_config);
    break;
  }
#endif
  default:
    LOG(FATAL) << "Unknown variant: " << static_cast<int>(config.variant);
  }

  LOG(INFO) << "Points A: " << points_a.size()
            << " Points B: " << points_b.size();
  double running_time = 0;

  for (int i = 0; i < config.repeat; i++) {
    auto& json_repeat = json_run["Repeat" + std::to_string(i)];

    if (config.execution == Execution::kCPU) {
      dist = hausdorff_distance->CalculateDistance(points_a, points_b);
    } else {
      dist =
          hausdorff_distance->CalculateDistance(stream, d_points_a, d_points_b);
    }
    auto repeat_stats = hausdorff_distance->get_stats();
    running_time += repeat_stats.at("ReportedTime").template get<double>();
    json_repeat.update(repeat_stats);
  }

  json_run["AvgTime"] = running_time / config.repeat;
  LOG(INFO) << "Avg Running Time " << json_run["AvgTime"] << " ms";

  stats.Log("HDResult", dist);

  if (config.check) {
    using hd_reference_impl = HausdorffDistanceEarlyBreak<COORD_T, N_DIMS>;
    auto& json_check = stats.Log("Check");
    typename hd_reference_impl::Config hd_config;

    hd_config.n_threads = std::thread::hardware_concurrency();
    auto hd_reference =
        std::make_unique<HausdorffDistanceEarlyBreak<COORD_T, N_DIMS>>(
            hd_config);
    auto answer_dist = hd_reference->CalculateDistance(points_a, points_b);
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
    } else {
      LOG(WARNING) << "Skip writting to JSON file " << config.json_file;
    }
  }
  return dist;
#endif

  return 0;
}
}  // namespace hd
