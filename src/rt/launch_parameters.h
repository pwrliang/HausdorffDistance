#ifndef RTSPATIAL_DETAILS_LAUNCH_PARAMETERS_H
#define RTSPATIAL_DETAILS_LAUNCH_PARAMETERS_H
#include "geoms/mbr.h"
#include "index/quantized_grid.h"
// #include "index/uniform_grid.h"
#include "utils/array_view.h"
#include "utils/queue.h"
#include "utils/type_traits.h"
#define RTSPATIAL_OPTIX_LAUNCH_PARAMS_NAME "params"

namespace hd {
namespace details {

template <typename COORD_T, int N_DIMS>
struct LaunchParamsNNUniformGrid {
  using point_t = typename cuda_vec<COORD_T, N_DIMS>::type;
  using mbr_t = Mbr<COORD_T, N_DIMS>;
  // using grid_t = dev::UniformGrid<COORD_T, N_DIMS>;

  ArrayView<uint32_t> in_queue;
  dev::Queue<uint32_t> term_queue;
  dev::Queue<uint32_t> miss_queue;
  ArrayView<point_t> points_a;
  ArrayView<point_t> points_b;
  ArrayView<mbr_t> mbrs_b;

  ArrayView<uint32_t> prefix_sum;
  ArrayView<uint32_t> point_b_ids;
  // grid_t grid;
  OptixTraversableHandle handle;
  COORD_T* cmax2;
  COORD_T radius;
  uint32_t* hit_counters;
  uint32_t* point_counters;
  uint32_t max_kcycles;
};

template <typename COORD_T, int N_DIMS>
struct LaunchParamsNNQuantizedGrid {
  using point_t = typename cuda_vec<COORD_T, N_DIMS>::type;
  using quantized_point_t = typename cuda_vec<int, N_DIMS>::type;
  using mbr_t = Mbr<COORD_T, N_DIMS>;

  struct Query {
    uint32_t point_id;
    uint32_t iter;
    Query() = default;
    DEV_HOST Query(uint32_t _point_id, uint32_t _iter)
        : point_id(_point_id), iter(_iter) {}
  };

  ArrayView<Query> in_queue;
  dev::Queue<uint32_t> miss_queue;
  dev::Queue<Query> hit_queue;
  dev::QuantizedGrid<COORD_T, N_DIMS> grid;
  ArrayView<point_t> points_a;
  ArrayView<point_t> points_b;
  ArrayView<quantized_point_t> representative_points;
  ArrayView<mbr_t> mbrs_b;
  ArrayView<uint32_t> prefix_sum;
  ArrayView<uint32_t> point_b_ids;
  OptixTraversableHandle handle;
  COORD_T* cmax2;
  COORD_T radius;
};
}  // namespace details

}  // namespace hd
#endif  // RTSPATIAL_DETAILS_LAUNCH_PARAMETERS_H
