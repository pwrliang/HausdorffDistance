#ifndef HAUSDORFF_DISTANCE_LBVH_H
#define HAUSDORFF_DISTANCE_LBVH_H
#include <flags.h>
#include <glog/logging.h>
#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <thrust/shuffle.h>

#include "distance.h"
#include "rt/reusable_buffer.h"
#include "rt/rt_engine.h"
#include "sampler.h"
#include "utils/bitset.h"
#include "utils/derived_atomic_functions.h"
#include "utils/helpers.h"
#include "utils/queue.h"
#include "utils/stopwatch.h"
#include "utils/stream.h"
#include "utils/type_traits.h"
#include "utils/util.h"

namespace hd {

template <typename COORD_T, int N_DIMS>
class HausdorffDistanceLBVH {
  using coord_t = COORD_T;
  using point_t = typename cuda_vec<COORD_T, N_DIMS>::type;

public:
 HausdorffDistanceLBVH() = default;


private:
  thrust::device_vector<point_t> points_a_;
  thrust::device_vector<point_t> points_b_;

};
}

#endif //HAUSDORFF_DISTANCE_LBVH_H
