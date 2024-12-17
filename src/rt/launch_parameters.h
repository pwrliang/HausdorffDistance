#ifndef RTSPATIAL_DETAILS_LAUNCH_PARAMETERS_H
#define RTSPATIAL_DETAILS_LAUNCH_PARAMETERS_H
#include "utils/array_view.h"
#include "utils/queue.h"
#include "utils/type_traits.h"
#define RTSPATIAL_OPTIX_LAUNCH_PARAMS_NAME "params"

namespace hd {
namespace details {

template <typename COORD_T, int N_DIMS>
struct LaunchParamsNN {
  using point_t = typename cuda_vec<COORD_T, N_DIMS>::type;

  dev::Queue<uint32_t> in_queue;
  dev::Queue<uint32_t> out_queue;
  ArrayView<point_t> points_a;
  ArrayView<point_t> points_b;
  ArrayView<OptixAabb> aabbs;
  OptixTraversableHandle handle;
  COORD_T *cmax2;
  uint32_t *total_hits;
};
}  // namespace details

}  // namespace hd
#endif  // RTSPATIAL_DETAILS_LAUNCH_PARAMETERS_H
