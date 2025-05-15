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
DECLARE_bool(move_to_origin);

// Execution/Algorithm
DECLARE_string(execution);
DECLARE_string(variant);
DECLARE_int32(parallelism);
DECLARE_int32(repeat);

// Parameters for algorithms
DECLARE_int32(seed);
// RT/Hybrid parameters
DECLARE_bool(auto_tune);
DECLARE_bool(fast_build_bvh);
DECLARE_bool(rebuild_bvh);
DECLARE_double(sample_rate);
DECLARE_double(max_hit_ratio);
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
DECLARE_string(max_hit_ratio_list);
DECLARE_string(n_points_cell_list);
#endif  // HAUSDORFF_DISTANCE_FLAGS_H
