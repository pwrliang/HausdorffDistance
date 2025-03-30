#ifndef HAUSDORFF_DISTANCE_RUN_CONFIG_H
#define HAUSDORFF_DISTANCE_RUN_CONFIG_H
#include <string>

#include "input_type.h"
namespace hd {
enum class Variant {
  kEARLY_BREAK,
  kZORDER,
  kYUAN,
  kRT,
  kHybrid,
  kBRANCH_BOUND,
  kITK
};

enum class Execution { kSerial, kParallel, kGPU };

struct RunConfig {
  std::string exec_path;
  std::string input_file1;
  std::string input_file2;
  std::string serialize_folder;
  std::string json_file;
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
  double move_offset;
  int repeat;
  // RT only
  double radius_step;
  bool sort_rays;
  bool fast_build_bvh;
  bool rebuild_bvh;
  double sample_rate;
  int max_hit;
  int max_reg_count;
  int n_points_cell;
};
}  // namespace hd

#endif  // HAUSDORFF_DISTANCE_RUN_CONFIG_H
