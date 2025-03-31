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
  double max_hit_reduce_factor;
  int max_reg_count;
  int n_points_cell;
  std::vector<float> radius_step_list;
  std::vector<float> sample_rate_list;
  std::vector<uint32_t> max_hit_list;
  std::vector<float> max_hit_reduce_factor_list;
  std::vector<uint32_t> n_points_cell_list;
  std::vector<bool> sort_rays_list;
  std::vector<bool> fast_build_bvh_list;
  std::vector<bool> rebuild_bvh_list;
};
}  // namespace hd

#endif  // HAUSDORFF_DISTANCE_RUN_CONFIG_H
