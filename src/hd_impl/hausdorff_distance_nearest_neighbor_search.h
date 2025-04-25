#ifndef HAUSDORFF_DISTANCE_NEAREST_NEIGHBOR_SEARCH_H
#define HAUSDORFF_DISTANCE_NEAREST_NEIGHBOR_SEARCH_H
#include <glog/logging.h>
#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <thrust/shuffle.h>

#include <sstream>

#include "cukd/builder.h"
#include "cukd/fcp.h"
#include "geoms/distance.h"
#include "geoms/mbr.h"
#include "hausdorff_distance.h"
#include "utils/array_view.h"
#include "utils/derived_atomic_functions.h"
#include "utils/helpers.h"
#include "utils/shared_value.h"
#include "utils/stopwatch.h"
#include "utils/stream.h"
#include "utils/type_traits.h"
#include "utils/util.h"

namespace hd {

template <typename COORD_T, int N_DIMS>
class HausdorffDistanceNearestNeighborSearch
    : public HausdorffDistance<COORD_T, N_DIMS> {
  using base_t = HausdorffDistance<COORD_T, N_DIMS>;
  using coord_t = COORD_T;
  using point_t = typename base_t::point_t;
  using data_traits = cukd::default_data_traits<point_t>;
  using kdtree_t = cukd::SpatialKDTree<point_t, data_traits>;

 public:
  coord_t CalculateDistance(const Stream& stream,
                            thrust::device_vector<point_t>& points_a,
                            thrust::device_vector<point_t>& points_b) override {
    kdtree_t tree_b;
    ArrayView<point_t> v_points_a(points_a);
    ArrayView<point_t> v_points_b(points_b);
    Stopwatch sw;
    auto& stats = this->stats_;

    sw.start();
    cukd::box_t<point_t>* d_bounds;
    cudaMallocManaged((void**) &d_bounds, sizeof(cukd::box_t<point_t>));

    cukd::buildTree<point_t, data_traits>(v_points_b.data(), v_points_b.size(),
                                          d_bounds, stream.cuda_stream());

    stream.Sync();
    sw.stop();
    stats["BuildKdTreeTime"] = sw.ms();

    sw.start();
    auto max2 = thrust::transform_reduce(
        thrust::cuda::par.on(stream.cuda_stream()), points_a.begin(),
        points_a.end(),
        [=] __device__(const point_t& p) mutable {
          auto point_b_id = cukd::cct::fcp<point_t, data_traits>(
              p, *d_bounds, v_points_b.data(), v_points_b.size());
          return EuclideanDistance2(p, v_points_b[point_b_id]);
        },
        0, thrust::maximum<coord_t>());

    cudaFreeAsync(d_bounds, stream.cuda_stream());
    stream.Sync();
    sw.stop();
    stats["SearchTreeTime"] = sw.ms();

    stats["Algorithm"] = "Nearest Neighbor Search";
    stats["Execution"] = "GPU";
    stats["ComparedPairs"] = points_a.size();

    return sqrt(max2);
  }
};
}  // namespace hd

#endif  // HAUSDORFF_DISTANCE_NEAREST_NEIGHBOR_SEARCH_H
