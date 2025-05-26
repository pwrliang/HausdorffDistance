#ifndef FEATURES_H
#define FEATURES_H
#include <cstdlib>
#include <nlohmann/json.hpp>
#include <vector>

namespace hd {

class Features {
  struct GridHisto {
    double P99Value;
    double P95Value;
    double P50Value;
    double P10Value;
    double P99Count;
    double P95Count;
    double P50Count;
    double P10Count;
    double GridSize[3];
  };
  /**
   * The order of the members must follow the generated decision tree
   */
  struct StaticFeatures {
    double Density;
    double NumPoints;
    double MBR_Lower[3];
    double MBR_Upper[3];
    double GiniIndex;
    GridHisto grid_histo;
    double NonEmptyCells;
    double TotalCells;
  };

  struct RuntimeFeatures {
    double HDLB;
    double HDUP;
  };

  struct AllFeatures {
    StaticFeatures static_features[2];
    GridHisto combined_grid_histo;
    RuntimeFeatures runtime_features;
  };

 public:
  Features() = default;

  explicit Features(int n_dims) : n_dims_(n_dims) {}

  void SetStaticFeatures(const nlohmann::json& json_input) {
    const auto& json_files = json_input.at("Files");

    memset(features_.static_features, sizeof(StaticFeatures) * 2, 0);
    memset(&features_.combined_grid_histo, sizeof(GridHisto), 0);

    for (int i = 0; i < 2; i++) {
      auto& input = features_.static_features[i];
      auto& json_file = json_files.at(i);

      input.Density = json_file.at("Density").get<double>();
      input.NumPoints = json_file.at("NumPoints").get<uint32_t>();

      for (int dim = 0; dim < n_dims_; dim++) {
        input.MBR_Lower[dim] = json_file.at("MBR")[dim].at("Lower");
        input.MBR_Upper[dim] = json_file.at("MBR")[dim].at("Upper");
      }

      auto& json_grid = json_file.at("Grid");

      input.GiniIndex = json_grid.at("GiniIndex");

      const auto& json_histo = json_grid.at("Histogram");
      auto& grid_histo = input.grid_histo;

      grid_histo.P99Value = -1;
      grid_histo.P95Value = -1;
      grid_histo.P50Value = -1;

      for (int bucket = json_histo.size() - 1; bucket >= 0; bucket--) {
        double count = json_histo[bucket].at("count");
        double percentile = json_histo[bucket].at("percentile");
        double value = json_histo[bucket].at("value");

        if (grid_histo.P99Value == -1 && percentile < 0.99) {
          grid_histo.P99Value = value;
          grid_histo.P99Count = count;
        } else if (grid_histo.P95Value == -1 && percentile < 0.95) {
          grid_histo.P95Value = value;
          grid_histo.P95Count = count;
        } else if (grid_histo.P50Value == -1 && percentile < 0.50) {
          grid_histo.P50Value = value;
          grid_histo.P50Count = count;
        } else if (grid_histo.P10Value == -1 && percentile < 0.10) {
          grid_histo.P10Value = value;
          grid_histo.P10Count = count;
        }
      }

      for (int dim = 0; dim < n_dims_; dim++) {
        grid_histo.GridSize[dim] = json_grid.at("GridSize")[dim];
      }
      input.NonEmptyCells = json_grid.at("NonEmptyCells");
      input.TotalCells = json_grid.at("TotalCells");
    }

    const auto& json_grid = json_input.at("Grid");
    const auto& json_histo = json_grid.at("Histogram");
    auto& grid_histo = features_.combined_grid_histo;

    grid_histo.P99Value = -1;
    grid_histo.P95Value = -1;
    grid_histo.P50Value = -1;

    for (int bucket = json_histo.size() - 1; bucket >= 0; bucket--) {
      double count = json_histo[bucket].at("count");
      double percentile = json_histo[bucket].at("percentile");
      double value = json_histo[bucket].at("value");

      if (grid_histo.P99Value == -1 && percentile < 0.99) {
        grid_histo.P99Value = value;
        grid_histo.P99Count = count;
      } else if (grid_histo.P95Value == -1 && percentile < 0.95) {
        grid_histo.P95Value = value;
        grid_histo.P95Count = count;
      } else if (grid_histo.P50Value == -1 && percentile < 0.50) {
        grid_histo.P50Value = value;
        grid_histo.P50Count = count;
      } else if (grid_histo.P10Value == -1 && percentile < 0.10) {
        grid_histo.P10Value = value;
        grid_histo.P10Count = count;
      }
    }

    for (int dim = 0; dim < n_dims_; dim++) {
      grid_histo.GridSize[dim] = json_grid.at("GridSize")[dim];
    }
  }

  void SetRuntimeFeatures(const nlohmann::json& json_repeat) {
    features_.runtime_features.HDLB =
        json_repeat.at("HDLowerBound").get<double>();
    features_.runtime_features.HDUP =
        json_repeat.at("HDUpperBound").get<double>();
  }

  std::vector<double> Serialize() const {
    auto n_doubles = sizeof(AllFeatures) / sizeof(double);
    std::vector<double> result(n_doubles);

    memcpy(result.data(), &features_, sizeof(AllFeatures));

    return result;
  }

 private:
  int n_dims_;
  AllFeatures features_;
};
}  // namespace hd
#endif  // FEATURES_H
