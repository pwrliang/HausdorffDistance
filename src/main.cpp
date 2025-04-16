#include <glog/logging.h>

#include "autotune_hausdorff_distance.cuh"
#include "flags.h"
#include "run_config.h"
#include "run_hausdorff_distance.cuh"

template <typename T>
std::vector<T> splitByComma(const std::string& input) {
  std::vector<T> result;
  std::stringstream ss(input);
  std::string token;

  while (std::getline(ss, token, ',')) {
    std::stringstream converter(token);
    T value;
    converter >> value;
    if (!converter.fail()) {
      result.push_back(value);
    } else {
      // Optionally handle error: skip or throw
      LOG(FATAL) << "Conversion failed for token: " << token;
    }
  }

  return result;
}

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
  using namespace hd;
  RunConfig config;
  std::string exec_path = argv[0];

  config.exec_path = exec_path.substr(0, exec_path.find_last_of("/"));
  config.input_file1 = FLAGS_input1;
  config.input_file2 = FLAGS_input2;
  config.stats_n_points_cell = FLAGS_stats_n_points_cell;
  config.serialize_folder = FLAGS_serialize;

  if (FLAGS_input_type == "wkt") {
    config.input_type = InputType::kWKT;
  } else if (FLAGS_input_type == "off") {
    config.input_type = InputType::kOFF;
  } else if (FLAGS_input_type == "image") {
    config.input_type = InputType::kImage;
  } else {
    LOG(FATAL) << "Unsupported input type: " << FLAGS_input_type;
  }

  if (FLAGS_variant == "eb") {
    config.variant = Variant::kEarlyBreak;
  } else if (FLAGS_variant == "zorder") {
    config.variant = Variant::kZORDER;
  } else if (FLAGS_variant == "yuan") {
    config.variant = Variant::kYUAN;
  } else if (FLAGS_variant == "rt") {
    config.variant = Variant::kRT;
  } else if (FLAGS_variant == "hybrid") {
    config.variant = Variant::kHybrid;
  } else if (FLAGS_variant == "branch-n-bound") {
    config.variant = Variant::kBRANCH_N_BOUND;
  } else if (FLAGS_variant == "nn") {
    config.variant = Variant::kNN;
  } else if (FLAGS_variant == "itk") {
    config.variant = Variant::kITK;
  } else {
    LOG(FATAL) << "Unknown variant: " << FLAGS_variant;
  }

  if (FLAGS_execution == "cpu") {
    config.execution = Execution::kCPU;
  } else if (FLAGS_execution == "gpu") {
    config.execution = Execution::kGPU;
  } else {
    LOG(FATAL) << "Unknown execution: " << FLAGS_execution;
  }

  config.parallelism = FLAGS_parallelism;
  config.seed = FLAGS_seed;
  config.check = FLAGS_check;
  config.n_dims = FLAGS_n_dims;
  config.is_double = FLAGS_is_double;
  config.limit = FLAGS_limit;
  config.move_offset = FLAGS_move_offset;
  config.repeat = FLAGS_repeat;
  config.auto_tune = FLAGS_auto_tune;
  config.radius_step = FLAGS_radius_step;
  config.sort_rays = FLAGS_sort_rays;
  config.fast_build_bvh = FLAGS_fast_build_bvh;
  config.rebuild_bvh = FLAGS_rebuild_bvh;
  config.sample_rate = FLAGS_sample_rate;
  config.max_hit = FLAGS_max_hit;
  config.max_reg_count = FLAGS_max_reg;
  config.n_points_cell = FLAGS_n_points_cell;
  config.json_file = FLAGS_json;
  config.overwrite = FLAGS_overwrite;

  CHECK(config.n_dims == 2 || config.n_dims == 3)
      << "Wrong number of dimensions, which can only be 2 or 3";

  if (FLAGS_vary_params) {
    config.radius_step_list = splitByComma<float>(FLAGS_radius_step_list);
    config.sample_rate_list = splitByComma<float>(FLAGS_sample_rate_list);
    config.max_hit_list = splitByComma<uint32_t>(FLAGS_max_hit_list);
    config.n_points_cell_list =
        splitByComma<uint32_t>(FLAGS_n_points_cell_list);
    if (config.sort_rays) {
      config.sort_rays_list.push_back(true);
    }
    config.sort_rays_list.push_back(false);
    if (config.fast_build_bvh) {
      config.fast_build_bvh_list.push_back(true);
    }
    config.fast_build_bvh_list.push_back(false);
    if (config.rebuild_bvh) {
      config.rebuild_bvh_list.push_back(true);
    }
    config.rebuild_bvh_list.push_back(false);

    CHECK_GT(config.radius_step_list.size(), 0);
    CHECK_GT(config.sample_rate_list.size(), 0);
    CHECK_GT(config.max_hit_list.size(), 0);
    CHECK_GT(config.n_points_cell_list.size(), 0);
    // hd::AutoTuneHausdorffDistance(config);
  } else {
    hd::RunHausdorffDistance(config);
  }

  google::ShutdownGoogleLogging();
}