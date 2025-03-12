#include <optix_function_table_definition.h>  // for g_optixFunctionTable

#include <algorithm>  // For std::shuffle
#include <cstdio>
#include <random>  // For random number generators

#include "hausdorff_distance_cpu.h"
#include "hausdorff_distance_gpu.h"
#include "hausdorff_distance_itk.h"
#include "hausdorff_distance_lbvh.h"
#include "hausdorff_distance_rt.h"
#include "img_loader.h"
#include "move_points.h"
#include "run_config.h"
#include "run_hausdorff_distance.cuh"
#include "utils/stopwatch.h"
#include "utils/type_traits.h"
#include "wkt_loader.h"

// TODO: Idea

/**
 * Build a grid over points B, each cell turns into cell.xyz +/- radius
 *
 * cast a ray from A to know a point intersects how many points from B
 *
 * for a given point A that has a high number of intersections -> early break
 * otherwise, use RT method
 *
 */
namespace hd {
template <typename COORD_T, int N_DIMS>
COORD_T RunHausdorffDistanceImpl(const RunConfig& config);

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
      // dist = RunHausdorffDistanceImpl<float, 3>(config);
    }
  }
  LOG(INFO) << "HausdorffDistance: distance is " << dist;
}

template <typename COORD_T, int N_DIMS>
COORD_T RunHausdorffDistanceImpl(const RunConfig& config) {
  using point_t = typename cuda_vec<COORD_T, N_DIMS>::type;
  std::vector<point_t> points_a, points_b;

  switch (config.input_type) {
  case InputType::kWKT: {
    points_a = LoadPoints<COORD_T, N_DIMS>(
        config.input_file1, config.serialize_folder, config.limit);
    points_b = LoadPoints<COORD_T, N_DIMS>(
        config.input_file2, config.serialize_folder, config.limit);
    break;
  }
  case InputType::kImage: {
    points_a = LoadImage<COORD_T, N_DIMS>(config.input_file1, config.limit);
    points_b = LoadImage<COORD_T, N_DIMS>(config.input_file2, config.limit);
    break;
  }
  }

  if (config.move_offset != 0) {
    MovePoints(points_a, points_b, config.move_offset);
  }

  COORD_T dist = -1;
  Stream stream;
  thrust::device_vector<point_t> d_points_a = points_a;
  thrust::device_vector<point_t> d_points_b = points_b;
  HausdorffDistanceRT<COORD_T, N_DIMS> hdist_rt;
  HausdorffDistanceLBVH<COORD_T, N_DIMS> hdist_lbvh;
  HausdorffDistanceRTConfig rt_config;
  std::string ptx_root = config.exec_path + "/ptx";

  rt_config.ptx_root = ptx_root.c_str();
  rt_config.rebuild_bvh = config.rebuild_bvh;
  rt_config.radius_step = config.radius_step;
  rt_config.init_radius = config.init_radius;
  rt_config.sample_rate = config.sample_rate;
  rt_config.max_hit = config.max_hit;

  hdist_rt.Init(rt_config);
  hdist_lbvh.SetPointsTo(stream, points_b.begin(), points_b.end());

  LOG(INFO) << "Points A: " << points_a.size()
            << " Points B: " << points_b.size();
  Stopwatch sw;
  double loading_time_ms = 0;

  sw.start();
  for (int i = 0; i < config.repeat; i++) {
    switch (config.variant) {
    case Variant::kEARLY_BREAK: {
      switch (config.execution) {
      case Execution::kSerial:
        dist = CalculateHausdorffDistance(points_a, points_b);
        break;
      case Execution::kParallel:
        dist = CalculateHausdorffDistanceParallel(points_a, points_b);
        break;
      case Execution::kGPU:
        dist = CalculateHausdorffDistanceGPU<point_t>(stream, d_points_a,
                                                      d_points_b);
        break;
      }
      break;
    }
    case Variant::kZORDER: {
      switch (config.execution) {
      case Execution::kSerial:
        dist = CalculateHausdorffDistanceZOrder(points_a, points_b);
        break;
      case Execution::kGPU:
        dist =
            CalculateHausdorffDistanceZorderGPU(stream, d_points_a, d_points_b);
        break;
      }
      break;
    }
    case Variant::kYUAN: {
      switch (config.execution) {
      case Execution::kSerial:
        dist = CalculateHausdorffDistanceYuan(points_a, points_b);
        break;
      }
      break;
    }
    case Variant::kRT: {
      dist = hdist_rt.CalculateDistance(stream, d_points_a, d_points_b);
      break;
    }
    case Variant::kHybrid: {
      dist = hdist_rt.CalculateDistanceHybrid(stream, d_points_a, d_points_b);
      break;
    }
    case Variant::kBRANCH_BOUND: {
      dist = hdist_lbvh.CalculateDistanceFrom(stream, points_a.begin(),
                                              points_a.end());
      break;
    }
    case Variant::kITK: {
      double curr_loading_time = 0;
      dist = CalculateHausdorffDistanceITK<N_DIMS>(config.input_file1.c_str(),
                                                   config.input_file2.c_str(),
                                                   curr_loading_time);
      loading_time_ms += curr_loading_time;
      break;
    }
    }
  }
  sw.stop();

  LOG(INFO) << "Running Time " << (sw.ms() - loading_time_ms) / config.repeat
            << " ms";

  if (config.check) {
    auto answer_dist = CalculateHausdorffDistanceParallel(points_a, points_b);
    auto diff = answer_dist - dist;

    if (dist != answer_dist) {
      LOG(FATAL) << std::fixed << std::setprecision(8)
                 << "Wrong HausdorffDistance. Result: " << dist
                 << " Answer: " << answer_dist << " Diff: " << diff;
    } else {
      LOG(INFO) << "HausdorffDistance is checked";
    }
  }

  return dist;
}
}  // namespace hd
