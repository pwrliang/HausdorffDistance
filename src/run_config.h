#ifndef HAUSDORFF_DISTANCE_RUN_CONFIG_H
#define HAUSDORFF_DISTANCE_RUN_CONFIG_H
#include <string>
enum class Variant { SERIAL, PARALLEL, GPU, RT, LBVH };

struct RunConfig {
  std::string exec_path;
  std::string input_file1;
  std::string input_file2;
  std::string serialize_folder;
  Variant variant;
  int n_dims;
  bool is_double;
  bool check;
  double move_offset;
};

#endif  // HAUSDORFF_DISTANCE_RUN_CONFIG_H
