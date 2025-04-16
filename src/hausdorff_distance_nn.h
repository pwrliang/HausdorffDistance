#ifndef HAUSDORFF_DISTANCE_NN_H
#define HAUSDORFF_DISTANCE_NN_H
#include <glog/logging.h>
#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <thrust/shuffle.h>

#include <queue>
#include <sstream>

#include "cukd/builder.h"
#include "cukd/fcp.h"
#include "distance.h"
#include "hd_bounds.h"
#include "mbr.h"
#include "utils/derived_atomic_functions.h"
#include "utils/helpers.h"
#include "utils/stopwatch.h"
#include "utils/stream.h"
#include "utils/type_traits.h"
#include "utils/util.h"

namespace hd {

template <typename COORD_T, int N_DIMS>
class HausdorffDistanceNN {
  using coord_t = COORD_T;
  using point_t = typename cuda_vec<COORD_T, N_DIMS>::type;
  using data_traits = cukd::default_data_traits<point_t>;

 public:
  COORD_T CalculateDistance(const Stream& stream,
                            thrust::device_vector<point_t>& points_a,
                            thrust::device_vector<point_t>& points_b) {
    using kdtree_t = cukd::SpatialKDTree<point_t, data_traits>;
    kdtree_t tree_a, tree_b;
    ArrayView<point_t> v_points_a(points_a);
    ArrayView<point_t> v_points_b(points_b);

    Stopwatch sw;

    sw.start();
    cukd::box_t<point_t>* d_bounds;
    cudaMallocManaged((void**) &d_bounds, sizeof(cukd::box_t<point_t>));

    cukd::buildTree<point_t, data_traits>(v_points_b.data(), v_points_b.size(),
                                          d_bounds, stream.cuda_stream());

    stream.Sync();
    sw.stop();
    LOG(INFO) << "Build Tree Time " << sw.ms();

    auto max2 = thrust::transform_reduce(
        thrust::cuda::par.on(stream.cuda_stream()), points_a.begin(),
        points_a.end(),
        [=] __device__(const point_t& p) mutable {
          auto point_b_id = cukd::cct::fcp<point_t, data_traits>(
              p, *d_bounds, v_points_b.data(), v_points_b.size());
          return EuclideanDistance2(p, v_points_b[point_b_id]);
        },
        0, thrust::maximum<COORD_T>());

    cudaFreeAsync(d_bounds, stream.cuda_stream());
    return sqrt(max2);
  }
};
}  // namespace hd

#endif  // HAUSDORFF_DISTANCE_NN_H
