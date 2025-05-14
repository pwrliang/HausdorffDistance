#ifndef FEATURES_H
#define FEATURES_H
#include <cstdlib>
#include <nlohmann/json.hpp>
#include <vector>

namespace hd {

template <int N_DIMS>
class FeaturesMaxHitInit {
  struct Input {
    double A_Density;
    double A_GiniIndex;
    double A_GridSize[3];
    double A_MaxPoints;
    double A_NonEmptyCells;
    double A_NumPoints;
    double B_Density;
    double B_GiniIndex;
    double B_GridSize[3];
    double B_MaxPoints;
    double B_NonEmptyCells;
    double B_NumPoints;
    double Density;
    double GiniIndex;
    double HDLowerBound;
    double HDUpperBound;
    double MaxPoints;
    double NonEmptyCells;
  };

 public:
  FeaturesMaxHitInit() = default;

  explicit FeaturesMaxHitInit(const nlohmann::json& json_Input) {
    auto& json_FileA = json_Input.at("FileA");
    auto& json_FileB = json_Input.at("FileB");
    auto& json_GridA = json_FileA.at("Grid");
    auto& json_GridB = json_FileB.at("Grid");

    input_.A_Density = json_FileA.at("Density").get<double>();
    input_.A_GiniIndex = json_GridA.at("GiniIndex").get<double>();
    for (int dim = 0; dim < N_DIMS; dim++) {
      input_.A_GridSize[dim] = json_GridA.at("GridSize")[dim];
    }
    input_.A_MaxPoints = json_GridA.at("MaxPoints").get<double>();
    input_.A_NonEmptyCells = json_GridA.at("NonEmptyCells").get<double>();
    input_.A_NumPoints = json_FileA.at("NumPoints").get<double>();

    input_.B_Density = json_FileB.at("Density").get<double>();
    input_.B_GiniIndex = json_GridB.at("GiniIndex").get<double>();
    for (int dim = 0; dim < N_DIMS; dim++) {
      input_.B_GridSize[dim] = json_GridB.at("GridSize")[dim];
    }
    input_.B_MaxPoints = json_GridB.at("MaxPoints").get<double>();
    input_.B_NonEmptyCells = json_GridB.at("NonEmptyCells").get<double>();
    input_.B_NumPoints = json_FileB.at("NumPoints").get<double>();
    input_.Density = json_Input.at("Density").get<double>();
  }

  void UpdateRunningInfo(const nlohmann::json& json_stats) {
    auto& json_Grid = json_stats.at("Grid");
    input_.GiniIndex = json_Grid.at("GiniIndex").get<double>();
    input_.HDLowerBound = json_stats.at("HDLowerBound").get<double>();
    input_.HDUpperBound = json_stats.at("HDUpperBound").get<double>();
    input_.MaxPoints = json_Grid.at("MaxPoints").get<double>();
    input_.NonEmptyCells = json_Grid.at("NonEmptyCells").get<double>();
  }

  std::vector<double> Serialize() const {
    auto n_doubles = sizeof(Input) / sizeof(double);
    std::vector<double> result(n_doubles + 2);
    for (int i = 0; i < n_doubles; i++) {
      result[i] = reinterpret_cast<const double*>(&input_)[i];
    }
    return result;
  }

 private:
  Input input_;
};

template <int N_DIMS>
class FeaturesMaxHitNext {
  struct Input {
    double A_Density;
    double A_GiniIndex;
    double A_GridSize[3];
    double A_MaxPoints;
    double A_NonEmptyCells;
    double A_NumPoints;
    double B_Density;
    double B_GiniIndex;
    double B_GridSize[3];
    double B_MaxPoints;
    double B_NonEmptyCells;
    double B_NumPoints;
    double CMax2;
    double ComparedPairs;
    double Density;
    double EBTime;
    double Hits1;
    double NumInputPoints;
    double NumOutputPoints;
    double NumTermPoints;
    double RTTime;
  };

 public:
  FeaturesMaxHitNext() = default;

  explicit FeaturesMaxHitNext(const nlohmann::json& json_Input) {
    auto& json_FileA = json_Input.at("FileA");
    auto& json_FileB = json_Input.at("FileB");
    auto& json_GridA = json_FileA.at("Grid");
    auto& json_GridB = json_FileB.at("Grid");

    input_.A_Density = json_FileA.at("Density").get<double>();
    input_.A_GiniIndex = json_GridA.at("GiniIndex").get<double>();
    for (int dim = 0; dim < N_DIMS; dim++) {
      input_.A_GridSize[dim] = json_GridA.at("GridSize")[dim];
    }
    input_.A_MaxPoints = json_GridA.at("MaxPoints").get<double>();
    input_.A_NonEmptyCells = json_GridA.at("NonEmptyCells").get<double>();
    input_.A_NumPoints = json_FileA.at("NumPoints").get<double>();

    input_.B_Density = json_FileB.at("Density").get<double>();
    input_.B_GiniIndex = json_GridB.at("GiniIndex").get<double>();
    for (int dim = 0; dim < N_DIMS; dim++) {
      input_.B_GridSize[dim] = json_GridB.at("GridSize")[dim];
    }
    input_.B_MaxPoints = json_GridB.at("MaxPoints").get<double>();
    input_.B_NonEmptyCells = json_GridB.at("NonEmptyCells").get<double>();
    input_.B_NumPoints = json_FileB.at("NumPoints").get<double>();
    input_.Density = json_Input.at("Density").get<double>();
  }

  void UpdateRunningInfo(const nlohmann::json& json_stats) {
    input_.CMax2 = json_stats.at("CMax2").get<double>();
    input_.ComparedPairs = json_stats.at("ComparedPairs").get<double>();
    input_.EBTime = json_stats.at("EBTime").get<double>();
    input_.Hits1 = json_stats.at("Hits").get<double>();
    input_.NumInputPoints = json_stats.at("NumInputPoints").get<double>();
    input_.NumOutputPoints = json_stats.at("NumOutputPoints").get<double>();
    input_.NumTermPoints = json_stats.at("NumTermPoints").get<double>();
    input_.RTTime = json_stats.at("RTTime").get<double>();
  }

  std::vector<double> Serialize() const {
    auto n_doubles = sizeof(Input) / sizeof(double);
    std::vector<double> result(n_doubles + 2);
    for (int i = 0; i < n_doubles; i++) {
      result[i] = reinterpret_cast<const double*>(&input_)[i];
    }
    return result;
  }

 private:
  Input input_;
};

template <int N_DIMS, int N_BUCKETS>
class FeaturesStatic {
  /**
   * The order of the members must follow the generated decision tree
   */
  struct Input {
    double GiniIndex;
    double GridSize[N_DIMS];
    double Histogram_count[N_BUCKETS];
    double Histogram_percentile[N_BUCKETS];
    double Histogram_value[N_BUCKETS];
    double MBR_Lower[N_DIMS];
    double MBR_Upper[N_DIMS];
    double MaxPoints;
    double NonEmptyCells;
    double NumPoints;
    double TotalCells;
  };

  static_assert(sizeof(Input) ==
                sizeof(double) * (5 + N_DIMS * 3 + N_BUCKETS * 3));

 public:
  FeaturesStatic() = default;

  explicit FeaturesStatic(const nlohmann::json& json_Input) {
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
    auto n_doubles = sizeof(Input) / sizeof(double);
    std::vector<double> result(n_doubles * 2 + 1);
    for (int i = 0; i < n_doubles; i++) {
      result[i] = reinterpret_cast<const double*>(&stats_A_)[i];
      result[i + n_doubles] = reinterpret_cast<const double*>(&stats_B_)[i];
    }
    return result;
  }

 private:
  Input stats_A_;
  Input stats_B_;

  void SetGridStats(const nlohmann::json& j, Input& stats) {
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

  void SetFileInfo(const nlohmann::json& j, Input& stats) {
    stats.NumPoints = j.at("NumPoints");
    for (int dim = 0; dim < N_DIMS; dim++) {
      stats.MBR_Lower[dim] = j.at("MBR")[dim].at("Lower");
      stats.MBR_Upper[dim] = j.at("MBR")[dim].at("Upper");
    }
  }
};
}  // namespace hd
#endif  // FEATURES_H
