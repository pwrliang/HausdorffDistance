#ifndef HAUSDORFF_DISTANCE_RUN_CONFIG_H
#define HAUSDORFF_DISTANCE_RUN_CONFIG_H
#include <string>
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

enum class InputType { kWKT, kImage };

struct RunConfig {
  std::string exec_path;
  std::string input_file1;
  std::string input_file2;
  std::string serialize_folder;
  InputType input_type;
  Variant variant;
  Execution execution;

  int parallelism;
  bool shuffle;
  bool check;
  int n_dims;
  bool is_double;
  int limit;
  double move_offset;
  int repeat;
  // RT only
  double radius_step;
  bool rebuild_bvh;
  double init_radius;
  double sample_rate;
  int max_hit;
  bool tensor;
  int triangle;
};

#endif  // HAUSDORFF_DISTANCE_RUN_CONFIG_H
