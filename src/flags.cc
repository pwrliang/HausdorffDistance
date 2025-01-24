#include "flags.h"

DEFINE_string(input1, "", "path of the point dataset1");
DEFINE_string(input2, "", "path of the point dataset2");
DEFINE_string(input_type, "wkt", "Input type can be 'wkt' or 'image'");
DEFINE_string(serialize, "", "a directory to store serialized point file");
DEFINE_string(variant, "eb", "can be the following options 'eb', 'zorder', 'yuan', 'rt', 'branch-bound'");
DEFINE_string(execution, "serial", "serial, parallel, or gpu");
DEFINE_int32(parallelism, -1, "How many cores to use for the parallel execution");
DEFINE_bool(shuffle, false, "Shuffle points (for eb/rt only)");
DEFINE_bool(check, true, "check correctness");
DEFINE_int32(n_dims, 0, "number of dimensions, which can be 2 or 3");
DEFINE_bool(is_double, false, "whether the dataset is a double");
DEFINE_int32(limit, INT32_MAX, "limit how many point to calculate");
DEFINE_double(move_offset, 0, "Move points from dataset2 by a given offset");
DEFINE_int32(repeat, 5, "Number of repeat to evaluate");
DEFINE_double(radius_step, 2, "Step of radius increase (RT only)");
DEFINE_bool(rebuild_bvh, false, "rebuild BVH (RT only)");
DEFINE_int32(raymulticast, 1, "Parallelism for casting rays (RT only)");
DEFINE_bool(nf, false, "Use near-far optimization");
DEFINE_int32(grid, 1024, "Grid size");