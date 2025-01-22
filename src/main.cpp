#include <glog/logging.h>

#include "flags.h"
#include "run_config.h"
#include "run_hausdorff_distance.cuh"

int main(int argc, char* argv[]) {
  FLAGS_stderrthreshold = 0;

  gflags::SetUsageMessage("Usage: -poly1 -poly2");
  if (argc == 1) {
    gflags::ShowUsageWithFlags(argv[0]);
    exit(1);
  }
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  gflags::ShutDownCommandLineFlags();

  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();

  RunConfig config;
  std::string exec_path = argv[0];

  config.exec_path = exec_path.substr(0, exec_path.find_last_of("/"));
  config.input_file1 = FLAGS_input1;
  config.input_file2 = FLAGS_input2;
  config.serialize_folder = FLAGS_serialize;

  if (FLAGS_variant == "eb") {
    config.variant = Variant::EARLY_BREAK;
  } else if (FLAGS_variant == "zorder") {
    config.variant = Variant::ZORDER;
  } else if (FLAGS_variant == "yuan") {
    config.variant = Variant::YUAN;
  } else if (FLAGS_variant == "rt") {
    config.variant = Variant::RT;
  } else if (FLAGS_variant == "branch-bound") {
    config.variant = Variant::BRANCH_BOUND;
  } else {
    LOG(FATAL) << "Unknown variant: " << FLAGS_variant;
  }

  if (FLAGS_execution == "serial") {
    config.execution = Execution::Serial;
  } else if (FLAGS_execution == "parallel") {
    config.execution = Execution::Parallel;
  } else if (FLAGS_execution == "gpu") {
    config.execution = Execution::GPU;
  } else {
    LOG(FATAL) << "Unknown execution: " << FLAGS_execution;
  }

  config.parallelism = FLAGS_parallelism;
  config.shuffle = FLAGS_shuffle;
  config.check = FLAGS_check;
  config.n_dims = FLAGS_n_dims;
  config.is_double = FLAGS_is_double;
  config.limit = FLAGS_limit;
  config.move_offset = FLAGS_move_offset;
  config.repeat = FLAGS_repeat;
  config.radius_step = FLAGS_radius_step;
  config.rebuild_bvh = FLAGS_rebuild_bvh;
  config.ray_multicast = FLAGS_raymulticast;
  config.nf = FLAGS_nf;
  config.grid_size = FLAGS_grid;
  hd::RunHausdorffDistance(config);

  google::ShutdownGoogleLogging();
}