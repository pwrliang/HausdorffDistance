#ifndef HAUSDORFF_DISTANCE_FLAGS_H
#define HAUSDORFF_DISTANCE_FLAGS_H
#include <gflags/gflags.h>

#include "flags.h"

DECLARE_string(input1);
DECLARE_string(input2);
DECLARE_string(input_type);
DECLARE_string(serialize);
DECLARE_string(variant);
DECLARE_string(execution);
DECLARE_int32(parallelism);
DECLARE_bool(shuffle);
DECLARE_bool(check);
DECLARE_int32(n_dims);
DECLARE_bool(is_double);
DECLARE_int32(limit);
DECLARE_double(move_offset);
DECLARE_int32(repeat);
DECLARE_double(radius_step);
DECLARE_bool(rebuild_bvh);
DECLARE_int32(raymulticast);
DECLARE_bool(nf);
DECLARE_int32(grid);
DECLARE_bool(auto_grid);
DECLARE_int32(batch);
DECLARE_int32(seed);
DECLARE_int32(distribution);
DECLARE_bool(bothSide);
#endif  // HAUSDORFF_DISTANCE_FLAGS_H
