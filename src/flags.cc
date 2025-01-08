#include "flags.h"

DEFINE_string(input1, "", "path of the point dataset1");
DEFINE_string(input2, "", "path of the point dataset2");
DEFINE_string(serialize, "", "a directory to store serialized point file");
DEFINE_string(variant, "serial", "serial/parallel/gpu/rt/lbvh");
DEFINE_bool(check, true, "check correctness");
DEFINE_int32(n_dims, 2, "number of dimensions, which can be 2 or 3");
DEFINE_bool(is_double, false, "whether the dataset is a double");
DEFINE_int32(limit, INT32_MAX, "limit how many point to calculate");
DEFINE_double(move_offset, 0, "Move points from dataset2 by a given offset");
DEFINE_int32(repeat, 5, "Number of repeat to evaluate");
DEFINE_double(radius, 0, "Init radius");
DEFINE_bool(cull, false, "Cull points");