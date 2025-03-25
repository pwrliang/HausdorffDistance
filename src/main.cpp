#include <glog/logging.h>

#include "flags.h"
#include "play.cuh"
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

  if (FLAGS_input_type == "wkt") {
    config.input_type = InputType::kWKT;
  } else if (FLAGS_input_type == "image") {
    config.input_type = InputType::kImage;
  } else {
    LOG(FATAL) << "Unsupported input type: " << FLAGS_input_type;
  }

  if (FLAGS_variant == "eb") {
    config.variant = Variant::kEARLY_BREAK;
  } else if (FLAGS_variant == "zorder") {
    config.variant = Variant::kZORDER;
  } else if (FLAGS_variant == "yuan") {
    config.variant = Variant::kYUAN;
  } else if (FLAGS_variant == "rt") {
    config.variant = Variant::kRT;
  } else if (FLAGS_variant == "hybrid") {
    config.variant = Variant::kHybrid;
  } else if (FLAGS_variant == "branch-bound") {
    config.variant = Variant::kBRANCH_BOUND;
  } else if (FLAGS_variant == "itk") {
    config.variant = Variant::kITK;
  } else {
    LOG(FATAL) << "Unknown variant: " << FLAGS_variant;
  }

  if (FLAGS_execution == "serial") {
    config.execution = Execution::kSerial;
  } else if (FLAGS_execution == "parallel") {
    config.execution = Execution::kParallel;
  } else if (FLAGS_execution == "gpu") {
    config.execution = Execution::kGPU;
  } else {
    LOG(FATAL) << "Unknown execution: " << FLAGS_execution;
  }

  config.parallelism = FLAGS_parallelism;
  config.check = FLAGS_check;
  config.n_dims = FLAGS_n_dims;
  config.is_double = FLAGS_is_double;
  config.limit = FLAGS_limit;
  config.move_offset = FLAGS_move_offset;
  config.repeat = FLAGS_repeat;
  config.radius_step = FLAGS_radius_step;
  config.rebuild_bvh = FLAGS_rebuild_bvh;
  config.init_radius = FLAGS_init_radius;
  config.sample_rate = FLAGS_sample_rate;
  config.max_hit = FLAGS_max_hit;
  config.tensor = FLAGS_tensor;
  config.triangle = FLAGS_triangle;

  CHECK(config.n_dims == 2 || config.n_dims == 3)
      << "Wrong number of dimensions, which can only be 2 or 3";

  hd::RunHausdorffDistance(config);

  google::ShutdownGoogleLogging();
}