#ifndef FEATURES_H
#define FEATURES_H
#include <cstdlib>
#include <nlohmann/json.hpp>
#include <vector>

namespace hd {
/**
 * The order of the members must follow the generated decision tree
 */
template <int N_DIMS, int N_BUCKETS>
struct FileStats {
  double MaxPoints;
  double GiniIndex;
  double GridSize[N_DIMS];
  double Histogram_count[N_BUCKETS];
  double Histogram_percentile[N_BUCKETS];
  double Histogram_value[N_BUCKETS];
  double MBR_Lower[N_DIMS];
  double MBR_Upper[N_DIMS];
  double NonEmptyCells;
  double NumPoints;
  double TotalCells;
};

template <int N_DIMS, int N_BUCKETS>
class Features {
  using file_stats_t = FileStats<N_DIMS, N_BUCKETS>;
  static_assert(sizeof(file_stats_t) ==
                sizeof(double) * (5 + N_DIMS * 3 + N_BUCKETS * 3));

 public:
  Features() = default;

  explicit Features(const nlohmann::json& json_Input) {
    auto& json_FileA = json_Input.at("FileA");
    auto& json_FileB = json_Input.at("FileB");
    auto& json_GridA = json_FileA.at("Grid");
    auto& json_GridB = json_FileB.at("Grid");

    SetFileInfo(json_FileA, stats_A_);
    SetGridStats(json_GridA, stats_A_);
    SetFileInfo(json_FileB, stats_B_);
    SetGridStats(json_GridB, stats_B_);
  }

  std::vector<double> Serialize() const {
    auto n_doubles = sizeof(file_stats_t) / sizeof(double);
    std::vector<double> result(n_doubles * 2);
    for (int i = 0; i < n_doubles; i++) {
      result[i] = reinterpret_cast<const double*>(&stats_A_)[i];
      result[i + n_doubles] = reinterpret_cast<const double*>(&stats_B_)[i];
    }
    return result;
  }

 private:
  file_stats_t stats_A_;
  file_stats_t stats_B_;

  void SetGridStats(const nlohmann::json& j, file_stats_t& stats) {
    stats.GiniIndex = j.at("GiniIndex");
    for (int dim = 0; dim < N_DIMS; dim++) {
      stats.GridSize[dim] = j.at("GridSize")[dim];
    }
    for (int bucket = 0; bucket < N_BUCKETS; bucket++) {
      double count;
      double percentile;
      double value;
      if (bucket < j.at("Histogram").size()) {
        count = j.at("Histogram")[bucket].at("count");
        percentile = j.at("Histogram")[bucket].at("percentile");
        value = j.at("Histogram")[bucket].at("value");
      } else {
        count = 0;
        percentile = 0;
        value = 0;
      }
      stats.Histogram_count[bucket] = count;
      stats.Histogram_percentile[bucket] = percentile;
      stats.Histogram_value[bucket] = value;
    }
    stats.MaxPoints = j.at("MaxPoints");
    stats.NonEmptyCells = j.at("NonEmptyCells");
    stats.TotalCells = j.at("TotalCells");
  }

  void SetFileInfo(const nlohmann::json& j, file_stats_t& stats) {
    stats.NumPoints = j.at("NumPoints");
    for (int dim = 0; dim < N_DIMS; dim++) {
      stats.MBR_Lower[dim] = j.at("MBR")[dim].at("Lower");
      stats.MBR_Upper[dim] = j.at("MBR")[dim].at("Upper");
    }
  }
};
}  // namespace hd
#endif  // FEATURES_H
