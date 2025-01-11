#include <algorithm>  // For std::shuffle
#include <cstdio>
#include <random>  // For random number generators

#include "flags.h"
#include "hausdorff_distance_gpu.h"
#include "hausdorff_distance_lbvh.h"
#include "hausdorff_distance_rt.h"
#include "move_points.h"
#include "run_config.h"
#include "run_hausdorff_distance.cuh"
#include "utils/stopwatch.h"
#include "utils/type_traits.h"
#include "wkt_loader.h"

namespace hd {
template <typename COORD_T, int N_DIMS>
COORD_T RunHausdorffDistanceImpl(const RunConfig& config);

void RunHausdorffDistance(const RunConfig& config) {
  double dist = -1;

  if (config.is_double) {
    if (config.n_dims == 2) {
      dist = RunHausdorffDistanceImpl<double, 2>(config);
    } else if (config.n_dims == 3) {
      dist = RunHausdorffDistanceImpl<double, 3>(config);
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

template <typename POINT_T>
typename vec_info<POINT_T>::type CalculateHausdorffDistance(
    std::vector<POINT_T>& points_a, std::vector<POINT_T>& points_b) {
  using coord_t = typename vec_info<POINT_T>::type;
  coord_t cmax = 0;
  std::random_device rd;  // Seed source
  std::mt19937 g(rd());   // Mersenne Twister engine seeded with rd()

  // Shuffle the vector
  std::shuffle(points_a.begin(), points_a.end(), g);
  std::shuffle(points_b.begin(), points_b.end(), g);
  uint32_t compared_pairs = 0;

  for (size_t i = 0; i < points_a.size(); i++) {
    coord_t cmin = std::numeric_limits<coord_t>::max();
    for (size_t j = 0; j < points_b.size(); j++) {
      auto d = EuclideanDistance2(points_a[i], points_b[j]);
      if (d < cmin) {
        cmin = d;
      }
      compared_pairs++;
      if (cmin <= cmax) {
        break;
      }
    }
    if (cmin != std::numeric_limits<coord_t>::max() && cmin > cmax) {
      cmax = cmin;
    }
  }
  LOG(INFO) << "Compared Pairs: " << compared_pairs;
  return sqrt(cmax);
}

template <typename T>
void update_maximum(std::atomic<T>& maximum_value, T const& value) noexcept {
  T prev_value = maximum_value;
  while (prev_value < value &&
         !maximum_value.compare_exchange_weak(prev_value, value)) {}
}

template <typename POINT_T>
typename vec_info<POINT_T>::type CalculateHausdorffDistanceParallel(
    std::vector<POINT_T>& points_a, std::vector<POINT_T>& points_b) {
  using coord_t = typename vec_info<POINT_T>::type;
  std::random_device rd;  // Seed source
  std::mt19937 g(rd());   // Mersenne Twister engine seeded with rd()

  // Shuffle the vector
  std::shuffle(points_a.begin(), points_a.end(), g);
  std::shuffle(points_b.begin(), points_b.end(), g);

  std::vector<std::thread> threads;
  auto thread_count = std::thread::hardware_concurrency();
  auto avg_points = (points_a.size() + thread_count - 1) / thread_count;
  std::atomic<coord_t> cmax;

  cmax = 0;

  for (int tid = 0; tid < thread_count; tid++) {
    threads.emplace_back(std::thread([&, tid]() {
      auto begin = tid * avg_points;
      auto end = std::min(begin + avg_points, points_a.size());

      for (int i = begin; i < end; i++) {
        auto cmin = std::numeric_limits<coord_t>::max();
        for (size_t j = 0; j < points_b.size(); j++) {
          auto d = EuclideanDistance2(points_a[i], points_b[j]);
          if (d < cmin) {
            cmin = d;
          }
          if (cmin < cmax) {
            break;
          }
        }
        if (cmin != std::numeric_limits<coord_t>::max()) {
          update_maximum(cmax, cmin);
        }
      }
    }));
  }

  for (auto& thread : threads) {
    thread.join();
  }

  return sqrt(cmax);
}

template <typename COORD_T, int N_DIMS>
COORD_T RunHausdorffDistanceImpl(const RunConfig& config) {
  using point_t = typename cuda_vec<COORD_T, N_DIMS>::type;
  auto points_a = LoadPoints<COORD_T, N_DIMS>(
      config.input_file1, config.serialize_folder, FLAGS_limit);
  auto points_b = LoadPoints<COORD_T, N_DIMS>(
      config.input_file2, config.serialize_folder, FLAGS_limit);

  if (config.move_offset != 0) {
    MovePoints(points_a, points_b, config.move_offset);
  }

  COORD_T dist;
  int n_repeat = FLAGS_repeat;
  Stream stream;
  thrust::device_vector<point_t> d_points_a = points_a;
  thrust::device_vector<point_t> d_points_b = points_b;
  HausdorffDistanceRT<COORD_T, N_DIMS> hdist_rt;
  HausdorffDistanceLBVH<COORD_T, N_DIMS> hdist_lbvh;
  HausdorffDistanceRTConfig rt_config;
  std::string ptx_root = config.exec_path + "/ptx";

  rt_config.ptx_root = ptx_root.c_str();
  rt_config.cull = FLAGS_cull;
  hdist_rt.Init(rt_config);
  hdist_rt.SetPointsTo(stream, points_b.begin(), points_b.end());

  hdist_lbvh.SetPointsTo(stream, points_b.begin(), points_b.end());

  LOG(INFO) << "Points A: " << points_a.size()
            << " Points B: " << points_b.size();
  Stopwatch sw;
  sw.start();
  for (int i = 0; i < n_repeat; i++) {
    switch (config.variant) {
    case Variant::SERIAL: {
      dist = CalculateHausdorffDistance(points_a, points_b);
      break;
    }
    case Variant::PARALLEL: {
      dist = CalculateHausdorffDistanceParallel(points_a, points_b);
      break;
    }
    case Variant::GPU: {
      dist = CalculateHausdorffDistanceGPU<point_t>(stream, d_points_a,
                                                    d_points_b);
      break;
    }
    case Variant::RT: {
      dist = hdist_rt.CalculateDistanceFrom(stream, points_a.begin(),
                                            points_a.end());
      break;
    }
    case Variant::LBVH: {
      dist = hdist_lbvh.CalculateDistanceFrom(stream, points_a.begin(),
                                              points_a.end());
      break;
    }
    }
  }
  sw.stop();

  LOG(INFO) << "Running Time " << sw.ms() / n_repeat << " ms";

  if (config.check) {
    auto answer_dist = CalculateHausdorffDistanceParallel(points_a, points_b);
    auto diff = answer_dist - dist;

    if (dist != answer_dist) {
      LOG(FATAL) << "Wrong HausdorffDistance. Result: " << dist
                 << " Answer: " << answer_dist << " Diff: " << diff;
    } else {
      LOG(INFO) << "HausdorffDistance is checked";
    }
  }

  return dist;
}
}  // namespace hd
