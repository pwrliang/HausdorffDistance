#ifndef RTSPATIAL_DETAILS_LAUNCH_PARAMETERS_H
#define RTSPATIAL_DETAILS_LAUNCH_PARAMETERS_H
#include "grid.h"
#include "mbr.h"
#include "utils/array_view.h"
#include "utils/bitset.h"
#include "utils/queue.h"
#include "utils/type_traits.h"
#define RTSPATIAL_OPTIX_LAUNCH_PARAMS_NAME "params"

namespace hd {
namespace details {

template <typename COORD_T, int N_DIMS>
struct LaunchParamsNN {
  using point_t = typename cuda_vec<COORD_T, N_DIMS>::type;

  ArrayView<uint32_t> in_queue;
  dev::Queue<uint32_t> term_queue;
  dev::Queue<uint32_t> miss_queue;
  ArrayView<point_t> points_a;
  ArrayView<point_t> points_b;
  OptixTraversableHandle handle;
  COORD_T* cmax2;
  COORD_T radius;
  COORD_T max_t;
  uint32_t* n_hits;
  uint32_t* hits_counters;
  uint32_t max_hit;
};

template <typename COORD_T, int N_DIMS>
struct LaunchParamsNNCompress {
  using point_t = typename cuda_vec<COORD_T, N_DIMS>::type;
  using mbr_t = Mbr<COORD_T, N_DIMS>;

  ArrayView<uint32_t> in_queue;
  dev::Queue<uint32_t> term_queue;
  dev::Queue<uint32_t> miss_queue;
  point_t* points_a;
  point_t* points_b;
  mbr_t* mbrs_b;
  uint32_t* prefix_sum;
  uint32_t* point_b_ids;
  OptixTraversableHandle handle;
  COORD_T* cmax2;
  COORD_T radius;
  uint32_t* n_hits;
  uint32_t* hits_counters;
  uint32_t max_hit;
};

}  // namespace details

}  // namespace hd
#endif  // RTSPATIAL_DETAILS_LAUNCH_PARAMETERS_H
