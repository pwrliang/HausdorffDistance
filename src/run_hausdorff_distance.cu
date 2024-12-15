#include <algorithm>  // For std::shuffle
#include <random>     // For random number generators

#include "flags.h"
#include "hausdorff_distance.h"
#include "run_config.h"
#include "run_hausdorff_distance.cuh"
#include "utils/stopwatch.h"
#include "utils/type_traits.h"
#include "wkt_loader.h"
namespace hd {
template <typename COORD_T, int N_DIMS>
COORD_T RunAllHausdorffDistance(const std::string& ptx_root,
                                const std::string& input_file1,
                                const std::string& input_file2,
                                const std::string& serialize_folder);

void RunHausdorffDistance(const RunConfig& config) {
  LOG(INFO) << "RunHausdorffDistance";
  double dist;
  std::string ptx_root = config.exec_path + "/ptx";

  if (config.is_double) {
    if (config.n_dims == 2) {
      dist = RunAllHausdorffDistance<double, 2>(ptx_root, config.input_file1,
                                                config.input_file2,
                                                config.serialize_folder);
    } else if (config.n_dims == 3) {
      // dist = RunAllHausdorffDistance<double, 3>(ptx_root, config.input_file1,
      //                                        config.input_file2);
    }
  } else {
    if (config.n_dims == 2) {
      dist = RunAllHausdorffDistance<float, 2>(ptx_root, config.input_file1,
                                               config.input_file2,
                                               config.serialize_folder);
    } else if (config.n_dims == 3) {
      // dist = RunAllHausdorffDistance<float, 3>(ptx_root, config.input_file1,
      //                                       config.input_file2);
    }
  }

  LOG(INFO) << "RunHausdorffDistance: dist = " << dist;
}

template <typename POINT_T>
double CalculateHausdorffDistance(std::vector<POINT_T>& points_a,
                                  std::vector<POINT_T>& points_b) {
  double cmax = 0.0;
  std::random_device rd;  // Seed source
  std::mt19937 g(rd());   // Mersenne Twister engine seeded with rd()

  // Shuffle the vector
  std::shuffle(points_a.begin(), points_a.end(), g);
  std::shuffle(points_b.begin(), points_b.end(), g);

  for (size_t i = 0; i < points_a.size(); i++) {
    double cmin = DBL_MAX;
    for (size_t j = 0; j < points_b.size(); j++) {
      auto d = EuclideanDistance2(points_a[i], points_b[j]);
      if (d < cmin) {
        cmin = d;
      }
      if (cmin < cmax) {
        break;
      }
    }
    if (cmin != DBL_MAX && cmin > cmax) {
      cmax = cmin;
    }
  }
  return sqrt(cmax);
}

template <typename T>
void update_maximum(std::atomic<T>& maximum_value, T const& value) noexcept {
  T prev_value = maximum_value;
  while (prev_value < value &&
         !maximum_value.compare_exchange_weak(prev_value, value)) {}
}

template <typename POINT_T>
double CalculateHausdorffDistanceParallel(std::vector<POINT_T>& points_a,
                                          std::vector<POINT_T>& points_b) {
  std::random_device rd;  // Seed source
  std::mt19937 g(rd());   // Mersenne Twister engine seeded with rd()

  // Shuffle the vector
  std::shuffle(points_a.begin(), points_a.end(), g);
  std::shuffle(points_b.begin(), points_b.end(), g);

  std::vector<std::thread> threads;
  auto thread_count = std::thread::hardware_concurrency();
  auto avg_points = (points_a.size() + thread_count - 1) / thread_count;
  std::atomic<double> cmax;

  cmax = 0.0;

  for (int tid = 0; tid < thread_count; tid++) {
    threads.emplace_back(std::thread([&, tid]() {
      auto begin = tid * avg_points;
      auto end = std::min(begin + avg_points, points_a.size());

      for (int i = begin; i < end; i++) {
        double cmin = DBL_MAX;
        for (size_t j = 0; j < points_b.size(); j++) {
          auto d = EuclideanDistance2(points_a[i], points_b[j]);
          if (d < cmin) {
            cmin = d;
          }
          if (cmin < cmax) {
            break;
          }
        }
        if (cmin != DBL_MAX) {
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
COORD_T RunAllHausdorffDistance(const std::string& ptx_root,
                                const std::string& input_file1,
                                const std::string& input_file2,
                                const std::string& serialize_folder) {
  auto points_a =
      LoadPoints<COORD_T, N_DIMS>(input_file1, serialize_folder, FLAGS_limit);
  auto points_b =
      LoadPoints<COORD_T, N_DIMS>(input_file2, serialize_folder, FLAGS_limit);

  LOG(INFO) << "Points A: " << points_a.size()
            << " Points B: " << points_b.size();
  // {
  //   Stopwatch sw;
  //   sw.start();
  //   auto answer = CalculateHausdorffDistance(points_a, points_b);
  //   sw.stop();
  //   LOG(INFO) << "CPU HausdorffDistance " << answer << " Time: " << sw.ms()
  //             << " ms";
  // }

  {
    Stopwatch sw;
    sw.start();
    auto answer = CalculateHausdorffDistanceParallel(points_a, points_b);
    sw.stop();
    LOG(INFO) << "CPU Parallel HausdorffDistance " << answer
              << " Time: " << sw.ms() << " ms";
  }

  HausdorffDistanceConfig config;
  HausdorffDistance<COORD_T, N_DIMS> hdist;
  Stream stream;

  config.ptx_root = ptx_root.c_str();
  hdist.Init(config);
  hdist.SetPointsTo(stream, points_b.begin(), points_b.end());
  Stopwatch sw;
  sw.start();
  auto dist =
      hdist.CalculateDistanceFrom(stream, points_a.begin(), points_a.end());
  sw.stop();

  LOG(INFO) << "GPU HausdorffDistance: " << dist << " Time: " << sw.ms()
            << " ms";
  return dist;
}
}  // namespace hd
