#ifndef HAUSDORFF_DISTANCE_GPU_H
#define HAUSDORFF_DISTANCE_GPU_H
#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <thrust/shuffle.h>
#include <utils/type_traits.h>

#include <cub/cub.cuh>

#include "distance.h"
#include "flags.h"
#include "utils/array_view.h"
#include "utils/derived_atomic_functions.h"
#include "utils/launcher.h"
#include "utils/shared_value.h"
#include "utils/stream.h"

namespace hd {

template <typename POINT_T>
typename vec_info<POINT_T>::type CalculateHausdorffDistanceGPU(
    const Stream& stream, thrust::device_vector<POINT_T>& points_a,
    thrust::device_vector<POINT_T>& points_b) {
  using coord_t = typename vec_info<POINT_T>::type;
  thrust::default_random_engine g;
  SharedValue<coord_t> cmax;
  auto* p_cmax = cmax.data();
  ArrayView<POINT_T> v_points_a(points_a);
  ArrayView<POINT_T> v_points_b(points_b);

  cmax.set(stream.cuda_stream(), 0);

  thrust::shuffle(thrust::cuda::par.on(stream.cuda_stream()), points_a.begin(),
                  points_a.end(), g);

  thrust::shuffle(thrust::cuda::par.on(stream.cuda_stream()), points_b.begin(),
                  points_b.end(), g);

  LaunchKernel(stream, [=] __device__() {
    auto warp_id = TID_1D / WARP_SIZE;
    auto n_warps = TOTAL_THREADS_1D / WARP_SIZE;
    auto lane_id = threadIdx.x % WARP_SIZE;
    using WarpReduce = cub::WarpReduce<coord_t>;
    __shared__ typename WarpReduce::TempStorage
        temp_storage[MAX_BLOCK_SIZE / WARP_SIZE];

    for (auto i = warp_id; i < v_points_a.size(); i += n_warps) {
      const auto& point_a = v_points_a[i];
      auto size_b_roundup =
          div_round_up(v_points_b.size(), WARP_SIZE) * WARP_SIZE;
      coord_t cmin = std::numeric_limits<coord_t>::max();

      for (auto j = lane_id; j < size_b_roundup; j += WARP_SIZE) {
        coord_t d = std::numeric_limits<coord_t>::max();
        if (j < v_points_b.size()) {
          const auto& point_b = v_points_b[j];
          d = EuclideanDistance2(point_a, point_b);
        }

        auto agg_min = WarpReduce(temp_storage[warp_id]).Reduce(d, cub::Min());
        int early_break = 0;

        if (lane_id == 0) {
          cmin = std::min(cmin, agg_min);
          if (cmin <= atomicMax(p_cmax, 0)) {
            early_break = 1;
          }
        }

        early_break = __shfl_sync(0xFFFFFFFF, early_break, 0);
        if (early_break) {
          break;
        }
      }
      if (lane_id == 0 && cmin != std::numeric_limits<coord_t>::max()) {
        atomicMax(p_cmax, cmin);
      }
    }
  });

  return sqrt(cmax.get(stream.cuda_stream()));
}

}  // namespace hd
#endif  // HAUSDORFF_DISTANCE_GPU_H
