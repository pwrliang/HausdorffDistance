#ifndef RTSPATIAL_DETAILS_LAUNCH_PARAMETERS_H
#define RTSPATIAL_DETAILS_LAUNCH_PARAMETERS_H
#include "mbr.h"
#include "grid.h"
#include "utils/array_view.h"
#include "utils/queue.h"
#include "utils/bitset.h"
#include "utils/type_traits.h"
#define RTSPATIAL_OPTIX_LAUNCH_PARAMS_NAME "params"
// Total length of a Tensor cannot be grater than 64*4 bytes
#define TENSOR_2D_BATCH_SIZE (16) // 16*2 = 32 FLOATS

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
  uint32_t* n_hits;
  uint32_t max_hit;
};


template <typename COORD_T, int N_DIMS>
struct LaunchParamsNNTensor {
  using point_t = typename cuda_vec<COORD_T, N_DIMS>::type;

  ArrayView<uint32_t> in_queue;
  dev::Queue<uint32_t> term_queue;
  dev::Queue<uint32_t> miss_queue;
  ArrayView<point_t> points_a;
  ArrayView<point_t> points_b;
  ArrayView<point_t> points_a_batched; // in_queue * TENSOR_xD_BATCH_SIZE
  ArrayView<point_t> points_b_batched; // in_queue * TENSOR_xD_BATCH_SIZE
  OptixTraversableHandle handle;
  COORD_T* cmax2;
  COORD_T radius;
  uint32_t* n_hits;
  uint32_t max_hit;
};

}  // namespace details

}  // namespace hd
#endif  // RTSPATIAL_DETAILS_LAUNCH_PARAMETERS_H
