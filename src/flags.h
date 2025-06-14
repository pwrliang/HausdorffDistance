#ifndef HAUSDORFF_DISTANCE_FLAGS_H
#define HAUSDORFF_DISTANCE_FLAGS_H
#include <gflags/gflags.h>

#include "flags.h"
// Input
DECLARE_string(input1);
DECLARE_string(input2);
DECLARE_string(input_type);
DECLARE_int32(stats_n_points_cell);
DECLARE_string(serialize);
DECLARE_int32(n_dims);
DECLARE_bool(is_double);
DECLARE_int32(limit);

// Preprocess
DECLARE_double(translate);
DECLARE_bool(normalize);

// Execution/Algorithm
DECLARE_string(execution);
DECLARE_string(variant);
DECLARE_int32(parallelism);
DECLARE_int32(repeat);

// Parameters for algorithms
DECLARE_int32(seed);
// RT/Hybrid parameters
DECLARE_bool(auto_tune);
DECLARE_bool(auto_tune_eb_only_threshold);
DECLARE_bool(auto_tune_n_points_cell);
DECLARE_bool(auto_tune_max_hit);
DECLARE_bool(fast_build_bvh);
DECLARE_bool(rebuild_bvh);
DECLARE_bool(rt_prune);
DECLARE_bool(rt_eb);
DECLARE_double(sample_rate);
DECLARE_int32(eb_only_threshold);
DECLARE_int32(max_hit);
DECLARE_int32(max_reg);
DECLARE_int32(n_points_cell);
DECLARE_int32(bit_count);

// Output
DECLARE_string(json);
DECLARE_bool(overwrite);
DECLARE_bool(check);

// For experiments only
DECLARE_bool(vary_params);
DECLARE_string(sample_rate_list);
DECLARE_string(eb_only_threshold_list);
DECLARE_string(max_hit_list);
DECLARE_string(n_points_cell_list);
#endif  // HAUSDORFF_DISTANCE_FLAGS_H
