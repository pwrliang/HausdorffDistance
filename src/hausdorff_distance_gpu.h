#ifndef HAUSDORFF_DISTANCE_GPU_H
#define HAUSDORFF_DISTANCE_GPU_H
#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <thrust/shuffle.h>
#include <utils/type_traits.h>

#include "distance.h"
#include "utils/array_view.h"
#include "utils/derived_atomic_functions.h"
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
  ArrayView<POINT_T> v_points_b(points_b);

  cmax.set(stream.cuda_stream(), 0);

  thrust::shuffle(thrust::cuda::par.on(stream.cuda_stream()), points_a.begin(),
                  points_a.end(), g);

  thrust::shuffle(thrust::cuda::par.on(stream.cuda_stream()), points_b.begin(),
                  points_b.end(), g);

  thrust::for_each(thrust::cuda::par.on(stream.cuda_stream()), points_a.begin(),
                   points_a.end(),
                   [=] __device__(const POINT_T& point_a) mutable {
                     coord_t cmin = std::numeric_limits<coord_t>::max();

                     for (uint32_t j = 0; j < v_points_b.size(); j++) {
                       auto d = EuclideanDistance2(point_a, v_points_b[j]);
                       if (d < cmin) {
                         cmin = d;
                       }
                       if (cmin <= atomicMax(p_cmax, 0)) {
                         break;
                       }
                     }
                     if (cmin != std::numeric_limits<coord_t>::max()) {
                       atomicMax(p_cmax, cmin);
                     }
                   });

  return sqrt(cmax.get(stream.cuda_stream()));
}

}  // namespace hd
#endif  // HAUSDORFF_DISTANCE_GPU_H
