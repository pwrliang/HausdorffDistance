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
//#include "models/tree_samplerate_3d.h"
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

  stats.Log("DateTime", get_current_datetime_string());
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
  for (int i = 0;
       i < std::min(10ul, std::min(points_a.size(), points_b.size())); i++) {
    char sa[100], sb[100];
    if (N_DIMS == 2) {
      sprintf(sa, "%.6f,%.6f", points_a[i].x, points_a[i].y);
      sprintf(sb, "%.6f,%.6f", points_b[i].x, points_b[i].y);
    } else if (N_DIMS == 3) {
      auto* p_a = &points_a[i].x;
      auto* p_b = &points_b[i].x;
      sprintf(sa, "%.6f,%.6f,%.6f", points_a[i].x, points_a[i].y, p_a[2]);
      sprintf(sb, "%.6f,%.6f,%.6f", points_b[i].x, points_b[i].y, p_b[2]);
    }
    printf("%d A %s B %s\n", i, sa, sb);
  }
  thrust::device_vector<point_t> d_points_a = points_a, d_points_b = points_b;

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

  auto& json_run = stats.Log("Running");

  json_run["Seed"] = config.seed;
  json_run["FastBuildBVH"] = config.fast_build_bvh;
  json_run["RebuildBVH"] = config.rebuild_bvh;
  json_run["SampleRate"] = config.sample_rate;
  json_run["NumPointsPerCell"] = config.n_points_cell;
  json_run["MaxHit"] = config.max_hit;
  json_run["EBOnlyThreshold"] = config.eb_only_threshold;
  json_run["AutoTune"] = config.auto_tune;

  COORD_T dist = -1;

  std::unique_ptr<HausdorffDistance<COORD_T, N_DIMS>> hausdorff_distance;

  if (config.parallelism <= 0) {
    config.parallelism = std::thread::hardware_concurrency();
  }

  switch (config.variant) {
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
#if 1
  case Variant::kCompareMethods: {
    using hd_impl_t = HausdorffDistanceCompareMethods<COORD_T, N_DIMS>;
    typename hd_impl_t::Config hd_config;
    std::string ptx_root = config.exec_path + "/ptx";
    hd_config.seed = config.seed;
    hd_config.ptx_root = ptx_root.c_str();
    hd_config.fast_build = config.fast_build_bvh;
    hd_config.rebuild_bvh = config.rebuild_bvh;
    hd_config.n_points_cell = config.n_points_cell;
    hd_config.max_hit = config.max_hit;
    hd_config.prune = config.rt_prune;
    hd_config.eb = config.rt_eb;

    hausdorff_distance = std::make_unique<hd_impl_t>(hd_config);
    break;
  }
#endif
  case Variant::kHybrid: {
    using hd_impl_t = HausdorffDistanceHybrid<COORD_T, N_DIMS>;
    typename hd_impl_t::Config hd_config;
    std::string ptx_root = config.exec_path + "/ptx";

    hd_config.seed = config.seed;
    hd_config.ptx_root = ptx_root.c_str();
    hd_config.auto_tune = config.auto_tune;
    hd_config.auto_tune_eb_only_threshold = config.auto_tune_eb_only_threshold;
    hd_config.auto_tune_n_points_cell = config.auto_tune_n_points_cell;
    hd_config.auto_tune_max_hit = config.auto_tune_max_hit;
    hd_config.fast_build = config.fast_build_bvh;
    hd_config.rebuild_bvh = config.rebuild_bvh;
    hd_config.sample_rate = config.sample_rate;
    hd_config.n_points_cell = config.n_points_cell;
    hd_config.eb_only_threshold = config.eb_only_threshold;
    hd_config.max_hit = config.max_hit;

    hausdorff_distance = std::make_unique<hd_impl_t>(hd_config);
    break;
  }

  case Variant::kRT: {
    using hd_impl_t = HausdorffDistanceRayTracing<COORD_T, N_DIMS>;
    typename hd_impl_t::Config hd_config;
    std::string ptx_root = config.exec_path + "/ptx";

    hd_config.ptx_root = ptx_root.c_str();
    hd_config.fast_build = config.fast_build_bvh;
    hd_config.rebuild_bvh = config.rebuild_bvh;
    hd_config.n_points_cell = config.n_points_cell;
    hd_config.prune = config.rt_prune;
    hd_config.eb = config.rt_eb;

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
#endif
  default:
    LOG(FATAL) << "Unknown variant: " << static_cast<int>(config.variant);
  }

  LOG(INFO) << "Points A: " << points_a.size()
            << " Points B: " << points_b.size();
  double running_time = 0;

  auto& json_repeats = json_run["Repeats"];

  for (int i = 0; i < config.repeat; i++) {
    json_repeats.push_back(nlohmann::json());
    auto& json_repeat = json_repeats.back();

    json_repeat["Repeat"] = i + 1;

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

  return 0;
}
}  // namespace hd
