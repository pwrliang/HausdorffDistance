#ifndef RTSPATIAL_DETAILS_LAUNCH_PARAMETERS_H
#define RTSPATIAL_DETAILS_LAUNCH_PARAMETERS_H
#include "mbr.h"
#include "utils/array_view.h"
#include "utils/queue.h"
#include "utils/bitset.h"
#include "utils/type_traits.h"
#define RTSPATIAL_OPTIX_LAUNCH_PARAMS_NAME "params"

namespace hd {
namespace details {

template <typename COORD_T, int N_DIMS>
struct LaunchParamsNN {
  using point_t = typename cuda_vec<COORD_T, N_DIMS>::type;

  ArrayView<uint32_t> in_queue;
  dev::Queue<uint32_t> out_queue;
  ArrayView<point_t> points_a;
  ArrayView<point_t> points_b;
  OptixTraversableHandle handle;
  COORD_T* cmax2;
  COORD_T radius;
  uint32_t* n_hits;
  uint32_t* n_compared_pairs;
  uint32_t* skip_count;
  uint32_t* skip_total_idx;
};

template <typename COORD_T, int N_DIMS>
struct LaunchParamsNNMultiCast {
  using point_t = typename cuda_vec<COORD_T, N_DIMS>::type;

  ArrayView<uint32_t> in_queue;
  dev::Queue<uint32_t> out_queue;
  ArrayView<point_t> points_a;
  ArrayView<point_t> points_b;
  COORD_T* cmin2; // threadIdx.x as index
  uint32_t* thread_counters; // threadIdx.x as index

  OptixTraversableHandle handle;
  Mbr<COORD_T, N_DIMS> mbr;

  COORD_T* cmax2;
  COORD_T radius;
  uint32_t* n_hits;
  uint32_t* n_compared_pairs;
  uint32_t* skip_count;
  uint32_t* skip_total_idx;
};

template <typename COORD_T, int N_DIMS>
struct LaunchParamsCull {
  using point_t = typename cuda_vec<COORD_T, N_DIMS>::type;

  ArrayView<uint32_t> in_queue;
  dev::Queue<uint32_t> out_queue;
  ArrayView<point_t> points_a;
  ArrayView<point_t> points_b;
  OptixTraversableHandle handle;
  COORD_T radius;
};

struct LaunchParamsPlay {
  ArrayView<float3> points;
  OptixTraversableHandle handle;
};

}  // namespace details

}  // namespace hd
#endif  // RTSPATIAL_DETAILS_LAUNCH_PARAMETERS_H
