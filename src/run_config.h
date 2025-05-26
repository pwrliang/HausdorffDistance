#ifndef HAUSDORFF_DISTANCE_RUN_CONFIG_H
#define HAUSDORFF_DISTANCE_RUN_CONFIG_H
#include <string>

#include "input_type.h"
namespace hd {
enum class Variant {
  kCompareMethods,
  kEarlyBreak,
  kRT,
  kRT_HDIST,
  kNearestNeighborSearch,
  kHybrid,
  kBranchAndBound,
  kITK
};

enum class Execution { kCPU, kGPU };

struct RunConfig {
  std::string exec_path;
  std::string input_file1;
  std::string input_file2;
  uint32_t stats_n_points_cell;
  std::string serialize_folder;
  std::string json_file;
  bool overwrite;
  InputType input_type;
  Variant variant;
  Execution execution;

  int parallelism;
  int seed;
  bool shuffle;
  bool check;
  int n_dims;
  bool is_double;
  int limit;
  float translate;
  bool normalize;
  int repeat;
  // RT only
  bool auto_tune;
  bool auto_tune_eb_only_threshold;
  bool auto_tune_n_points_cell;
  bool auto_tune_max_hit;
  bool fast_build_bvh;
  bool rebuild_bvh;
  bool rt_prune;
  bool rt_eb;
  float sample_rate;
  uint32_t eb_only_threshold;
  uint32_t max_hit;
  int max_reg_count;
  int n_points_cell;
  int bit_count;
  std::vector<float> sample_rate_list;
  std::vector<uint32_t> eb_only_threshold_list;
  std::vector<uint32_t> max_hit_list;
  std::vector<uint32_t> n_points_cell_list;
};
}  // namespace hd

#endif  // HAUSDORFF_DISTANCE_RUN_CONFIG_H
