#ifndef HAUSDORFF_DISTANCE_RUN_CONFIG_H
#define HAUSDORFF_DISTANCE_RUN_CONFIG_H
#include <string>
enum class Variant { kEARLY_BREAK, kZORDER, kYUAN, kRT, kBRANCH_BOUND, kITK };

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
  int ray_multicast;
  bool nf;
  int grid_size;
  bool auto_grid;
};

#endif  // HAUSDORFF_DISTANCE_RUN_CONFIG_H
