#include "flags.h"

DEFINE_string(input1, "", "path of the point dataset1");
DEFINE_string(input2, "", "path of the point dataset2");
DEFINE_int32(n_dims, 2, "number of dimensions, which can be 2 or 3");
DEFINE_bool(is_double, false, "whether the dataset is a double");
