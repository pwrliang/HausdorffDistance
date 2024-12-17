#ifndef HAUSDORFF_DISTANCE_FLAGS_H
#define HAUSDORFF_DISTANCE_FLAGS_H
#include <gflags/gflags.h>

#include "flags.h"

DECLARE_string(input1);
DECLARE_string(input2);
DECLARE_string(serialize);
DECLARE_int32(n_dims);
DECLARE_bool(is_double);
DECLARE_int32(limit);
DECLARE_double(move_offset);
#endif  // HAUSDORFF_DISTANCE_FLAGS_H
