#include <glog/logging.h>

#include <algorithm>
#include <cctype>
#include <string>

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

  std::string input_type = FLAGS_input_type;
  std::transform(input_type.begin(), input_type.end(), input_type.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  if (input_type == "wkt") {
    config.input_type = InputType::kWKT;
  } else if (input_type == "off") {
    config.input_type = InputType::kOFF;
  } else if (input_type == "image") {
    config.input_type = InputType::kImage;
  } else if (input_type == "ply") {
    config.input_type = InputType::kPLY;
  } else {
    LOG(FATAL) << "Unsupported input type: " << FLAGS_input_type;
  }

  std::string variant = FLAGS_variant;
  std::transform(variant.begin(), variant.end(), variant.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  if (variant == "compare-methods") {
    config.variant = Variant::kCompareMethods;
  } else if (variant == "eb") {
    config.variant = Variant::kEarlyBreak;
  } else if (variant == "rt-hdist") {
    config.variant = Variant::kRT_HDIST;
  } else if (variant == "rt") {
    config.variant = Variant::kRT;
  } else if (variant == "hybrid") {
    config.variant = Variant::kHybrid;
  } else if (variant == "bnb") {
    config.variant = Variant::kBranchAndBound;
  } else if (variant == "nn") {
    config.variant = Variant::kNearestNeighborSearch;
  } else if (variant == "itk") {
    config.variant = Variant::kITK;
  } else {
    LOG(FATAL) << "Unknown variant: " << FLAGS_variant;
  }

  std::string execution = FLAGS_execution;
  std::transform(execution.begin(), execution.end(), execution.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  if (execution == "cpu") {
    config.execution = Execution::kCPU;
  } else if (execution == "gpu") {
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
  config.translate = FLAGS_translate;
  config.normalize = FLAGS_normalize;
  config.repeat = FLAGS_repeat;
  config.auto_tune = FLAGS_auto_tune;
  config.fast_build_bvh = FLAGS_fast_build_bvh;
  config.rebuild_bvh = FLAGS_rebuild_bvh;
  config.rt_prune = FLAGS_rt_prune;
  config.rt_eb = FLAGS_rt_eb;
  config.sample_rate = FLAGS_sample_rate;
  config.eb_only_threshold = FLAGS_eb_only_threshold;
  config.max_hit = FLAGS_max_hit;
  config.max_reg_count = FLAGS_max_reg;
  config.n_points_cell = FLAGS_n_points_cell;
  config.bit_count = FLAGS_bit_count;
  config.json_file = FLAGS_json;
  config.overwrite = FLAGS_overwrite;

  CHECK(config.n_dims == 2 || config.n_dims == 3)
      << "Wrong number of dimensions, which can only be 2 or 3";

  if (FLAGS_vary_params) {
    config.sample_rate_list = splitByComma<float>(FLAGS_sample_rate_list);
    config.max_hit_list = splitByComma<uint32_t>(FLAGS_max_hit_list);
    config.eb_only_threshold_list =
        splitByComma<uint32_t>(FLAGS_eb_only_threshold_list);
    config.n_points_cell_list =
        splitByComma<uint32_t>(FLAGS_n_points_cell_list);
    CHECK_GT(config.sample_rate_list.size(), 0);
    CHECK_GT(config.n_points_cell_list.size(), 0);
    hd::AutoTuneHausdorffDistance(config);
  } else {
    hd::RunHausdorffDistance(config);
  }

  google::ShutdownGoogleLogging();
}