#ifndef FEATURES_H
#define FEATURES_H
#include <cstdlib>
#include <nlohmann/json.hpp>
#include <vector>

namespace hd {

template <int N_BUCKETS>
class FeaturesStatic {
  /**
   * The order of the members must follow the generated decision tree
   */
  struct Input {
    double GiniIndex;
    double GridSize[3];
    double Histogram_count[N_BUCKETS];
    double Histogram_percentile[N_BUCKETS];
    double Histogram_value[N_BUCKETS];
    double MBR_Lower[3];
    double MBR_Upper[3];
    double MaxPoints;
    double NonEmptyCells;
    double NumPoints;
    double TotalCells;
  };

  static_assert(sizeof(Input) ==
                sizeof(double) * (5 + 3 * 3 + N_BUCKETS * 3));

 public:
  FeaturesStatic() = default;

  explicit FeaturesStatic(const nlohmann::json& json_Input, int n_dims) {
    auto& json_FileA = json_Input.at("FileA");
    auto& json_FileB = json_Input.at("FileB");
    auto& json_GridA = json_FileA.at("Grid");
    auto& json_GridB = json_FileB.at("Grid");

    memset(&stats_A_, sizeof(Input), 0);
    memset(&stats_B_, sizeof(Input), 0);

    SetFileInfo(json_FileA, stats_A_, n_dims);
    SetGridStats(json_GridA, stats_A_, n_dims);
    SetFileInfo(json_FileB, stats_B_, n_dims);
    SetGridStats(json_GridB, stats_B_,n_dims );
  }

  std::vector<double> Serialize() const {
    auto n_doubles = sizeof(Input) / sizeof(double);
    std::vector<double> result(n_doubles * 2);
    for (int i = 0; i < n_doubles; i++) {
      result[i] = reinterpret_cast<const double*>(&stats_A_)[i];
      result[i + n_doubles] = reinterpret_cast<const double*>(&stats_B_)[i];
    }
    return result;
  }

 private:
  Input stats_A_;
  Input stats_B_;

  void SetGridStats(const nlohmann::json& j, Input& stats, int n_dims) {
    stats.GiniIndex = j.at("GiniIndex");
    for (int dim = 0; dim < n_dims; dim++) {
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

  void SetFileInfo(const nlohmann::json& j, Input& stats, int n_dims) {
    stats.NumPoints = j.at("NumPoints");
    for (int dim = 0; dim < n_dims; dim++) {
      stats.MBR_Lower[dim] = j.at("MBR")[dim].at("Lower");
      stats.MBR_Upper[dim] = j.at("MBR")[dim].at("Upper");
    }
  }
};
}  // namespace hd
#endif  // FEATURES_H
