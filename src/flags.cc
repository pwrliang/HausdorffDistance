#include "flags.h"

DEFINE_string(input1, "", "path of the point dataset1");
DEFINE_string(input2, "", "path of the point dataset2");
DEFINE_string(input_type, "", "Input type can be 'wkt' or 'image'");
DEFINE_string(serialize, "", "a directory to store serialized point file");
DEFINE_string(json, "", "Output path of json file");
DEFINE_string(
    variant, "eb",
    "can be the following options 'eb', 'zorder', 'yuan', 'rt', 'hybrid', "
    "'branch-bound'");
DEFINE_string(execution, "serial", "serial, parallel, or gpu");
DEFINE_int32(parallelism, -1,
             "How many cores to use for the parallel execution");
DEFINE_int32(seed, 0, "Random number seed");
DEFINE_bool(check, true, "check correctness");
DEFINE_int32(n_dims, 0, "number of dimensions, which can be 2 or 3");
DEFINE_bool(is_double, false, "whether the dataset is a double");
DEFINE_int32(limit, INT32_MAX, "limit how many point to calculate");
DEFINE_double(move_offset, 0, "Move points from dataset2 by a given offset");
DEFINE_int32(repeat, 5, "Number of repeat to evaluate");
DEFINE_double(radius_step, 2, "Step of radius increase (RT only)");
DEFINE_bool(fast_build_bvh, false, "Prefer fast build BVH");
DEFINE_bool(rebuild_bvh, false, "rebuild BVH (RT only)");
DEFINE_bool(sort_rays, false, "sort rays by their Morton codes");
DEFINE_double(sample_rate, 0.001, "");
DEFINE_int32(max_hit, 128, "Max number of hit by RT method before using EB");
DEFINE_int32(max_reg, 0, "Max # of registers for RT");
DEFINE_int32(n_points_cell, 0, "Number of points per cell");
