#include "flags.h"
// Input
DEFINE_string(input1, "", "path of the point dataset1");
DEFINE_string(input2, "", "path of the point dataset2");
DEFINE_string(input_type, "", "Input type can be 'wkt' or 'image'");
DEFINE_int32(stats_n_points_cell, 10, "");
DEFINE_string(serialize, "", "a directory to store serialized point file");
DEFINE_int32(n_dims, 0, "number of dimensions, which can be 2 or 3");
DEFINE_bool(is_double, false, "whether the dataset is a double");
DEFINE_int32(limit, INT32_MAX, "limit how many point to calculate");

// Preprocess
DEFINE_double(translate, 0.0, "Move points from dataset2 by a given ratio");
DEFINE_bool(normalize, false, "Normalize points to [0, 1]");

// Execution/Algorithm
DEFINE_string(execution, "gpu", "cpu or gpu");
DEFINE_string(
    variant, "eb",
    "can be the following options 'eb', 'rt', 'hybrid', 'branch-bound'");
DEFINE_int32(parallelism, -1,
             "How many cores to use for the parallel execution");
DEFINE_int32(repeat, 5, "Number of repeat to evaluate");

// Parameters for algorithms
DEFINE_int32(seed, 0, "Random number seed");
// RT/Hybrid parameters
DEFINE_bool(auto_tune, false, "Automatic tuning parameters");
DEFINE_bool(auto_tune_eb_only_threshold, false, "");
DEFINE_bool(auto_tune_n_points_cell, false, "");
DEFINE_bool(auto_tune_max_hit, false, "");
DEFINE_bool(fast_build_bvh, false, "Prefer fast build BVH");
DEFINE_bool(rebuild_bvh, false, "rebuild BVH (RT only)");
DEFINE_bool(rt_prune, true, "prune with UB and LB");
DEFINE_bool(rt_eb, true, "use eb in RT");
DEFINE_double(sample_rate, 0.01, "");
DEFINE_int32(max_hit, 256,
             "The max numbers of hit to non-empty cells with RT "
             "method before using EB. If it is set to zero, we will try our "
             "best to find a suitable value");
DEFINE_int32(eb_only_threshold, 0, "Number of iters for using EB");
DEFINE_int32(max_reg, 0, "Max # of registers for RT");
DEFINE_int32(n_points_cell, 8, "Number of points per cell.");
DEFINE_int32(bit_count, 7, "Grid bit count setting of RT-HDIST");

// Output
DEFINE_string(json, "", "Output path of json file");
DEFINE_bool(overwrite, false, "Whether overwrite json file");
DEFINE_bool(check, true, "check correctness");

// For experiments only
DEFINE_bool(vary_params, false,
            "Varying parameters for producing training data");
DEFINE_string(sample_rate_list, "0.01", "");
DEFINE_string(eb_only_threshold_list, "0", "");
DEFINE_string(max_hit_list, "256", "");
DEFINE_string(n_points_cell_list, "8", "");

