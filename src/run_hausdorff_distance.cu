#include <algorithm>  // For std::shuffle
#include <random>     // For random number generators

#include "hausdorff_distance.h"
#include "run_config.h"
#include "run_hausdorff_distance.cuh"
#include "utils/stopwatch.h"
#include "utils/type_traits.h"
#include "wkt_loader.h"
namespace hd {
template <typename COORD_T, int N_DIMS>
COORD_T RunHausdorffDistance(const std::string& ptx_root,
                             const std::string& input_file1,
                             const std::string& input_file2);

void RunHausdorffDistance(const RunConfig& config) {
  LOG(INFO) << "RunHausdorffDistance";
  double dist;
  std::string ptx_root = config.exec_path + "/ptx";

  if (config.is_double) {
    if (config.n_dims == 2) {
      dist = RunHausdorffDistance<double, 2>(ptx_root, config.input_file1,
                                             config.input_file2);
    } else if (config.n_dims == 3) {
      // dist = RunHausdorffDistance<double, 3>(ptx_root, config.input_file1,
      //                                        config.input_file2);
    }
  } else {
    if (config.n_dims == 2) {
      dist = RunHausdorffDistance<float, 2>(ptx_root, config.input_file1,
                                            config.input_file2);
    } else if (config.n_dims == 3) {
      // dist = RunHausdorffDistance<float, 3>(ptx_root, config.input_file1,
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

template <typename COORD_T, int N_DIMS>
COORD_T RunHausdorffDistance(const std::string& ptx_root,
                             const std::string& input_file1,
                             const std::string& input_file2) {
  auto points_a = LoadPoints<COORD_T, N_DIMS>(input_file1);
  auto points_b = LoadPoints<COORD_T, N_DIMS>(input_file2);

  LOG(INFO) << "Points A: " << points_a.size()
            << " Points B: " << points_b.size();
  Stopwatch sw;
  sw.start();
  auto answer = CalculateHausdorffDistance(points_a, points_b);
  sw.stop();
  LOG(INFO) << "CPU HausdorffDistance " << answer << " Time: " << sw.ms()
            << " ms";

  HausdorffDistanceConfig config;
  HausdorffDistance<COORD_T, N_DIMS> hdist;
  Stream stream;

  config.ptx_root = ptx_root.c_str();
  hdist.Init(config);
  hdist.SetPointsTo(stream, points_b.begin(), points_b.end());

  sw.start();
  auto dist =
      hdist.CalculateDistanceFrom(stream, points_a.begin(), points_a.end());
  sw.stop();

  LOG(INFO) << "GPU HausdorffDistance: " << dist << " Time: " << sw.ms()
            << " ms";
  return dist;
}
}  // namespace hd
