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
  config.n_dims = FLAGS_n_dims;
  config.is_double = FLAGS_is_double;
  config.check = FLAGS_check;

  if (FLAGS_variant == "serial") {
    config.variant = Variant::SERIAL;
  } else if (FLAGS_variant == "parallel") {
    config.variant = Variant::PARALLEL;
  } else if (FLAGS_variant == "gpu") {
    config.variant = Variant::GPU;
  } else if (FLAGS_variant == "rt") {
    config.variant = Variant::RT;
  } else if (FLAGS_variant == "lbvh") {
    config.variant = Variant::LBVH;
  } else {
    LOG(FATAL) << "Unknown variant: " << FLAGS_variant;
  }

  hd::RunHausdorffDistance(config);

  google::ShutdownGoogleLogging();
}