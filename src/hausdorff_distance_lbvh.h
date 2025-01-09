#ifndef HAUSDORFF_DISTANCE_LBVH_H
#define HAUSDORFF_DISTANCE_LBVH_H
#include <glog/logging.h>
#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <thrust/shuffle.h>

#include <boost/mpl/print.hpp>
#include <queue>
#include <sstream>

#include "distance.h"
#include "hd_bounds.h"
#include "lbvh/lbvh.cuh"
#include "mbr.h"
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

namespace detail {
template <typename COORD_T, int N_DIMS>
struct aabb_getter {};

template <>
struct aabb_getter<float, 2> {
  __device__ lbvh::aabb<float> operator()(const float2& f) const noexcept {
    lbvh::aabb<float> retval;
    retval.upper.x = f.x;
    retval.upper.y = f.y;
    retval.upper.z = 0;
    retval.upper.w = 0;

    retval.lower.x = f.x;
    retval.lower.y = f.y;
    retval.lower.z = 0;
    retval.lower.w = 0;
    return retval;
  }
};

template <>
struct aabb_getter<float, 3> {
  __device__ lbvh::aabb<float> operator()(const float3& f) const noexcept {
    lbvh::aabb<float> retval;
    retval.upper.x = f.x;
    retval.upper.y = f.y;
    retval.upper.z = f.z;
    retval.upper.w = 0;

    retval.lower.x = f.x;
    retval.lower.y = f.y;
    retval.lower.z = f.z;
    retval.lower.w = 0;
    return retval;
  }
};

template <>
struct aabb_getter<double, 2> {
  __device__ lbvh::aabb<double> operator()(const double2& f) const noexcept {
    lbvh::aabb<double> retval;
    retval.upper.x = f.x;
    retval.upper.y = f.y;
    retval.upper.z = 0;
    retval.upper.w = 0;

    retval.lower.x = f.x;
    retval.lower.y = f.y;
    retval.lower.z = 0;
    retval.lower.w = 0;
    return retval;
  }
};

template <>
struct aabb_getter<double, 3> {
  __device__ lbvh::aabb<double> operator()(const double3& f) const noexcept {
    lbvh::aabb<double> retval;
    retval.upper.x = f.x;
    retval.upper.y = f.y;
    retval.upper.z = f.z;
    retval.upper.w = 0;

    retval.lower.x = f.x;
    retval.lower.y = f.y;
    retval.lower.z = f.z;
    retval.lower.w = 0;
    return retval;
  }
};

template <typename COORD_T, int N_DIMS>
struct AABBToMbr {};

template <typename COORD_T>
struct AABBToMbr<COORD_T, 2> {
  using point_t = typename Mbr<COORD_T, 2>::point_t;
  Mbr<COORD_T, 2> operator()(const lbvh::aabb<COORD_T>& aabb) const {
    point_t lower, upper;

    lower.x = aabb.lower.x;
    lower.y = aabb.lower.y;
    upper.x = aabb.upper.x;
    upper.y = aabb.upper.y;

    return Mbr<COORD_T, 2>(lower, upper);
  }
};

template <typename COORD_T>
struct AABBToMbr<COORD_T, 3> {
  using point_t = typename Mbr<COORD_T, 3>::point_t;
  Mbr<COORD_T, 2> operator()(const lbvh::aabb<COORD_T>& aabb) const {
    point_t lower, upper;

    lower.x = aabb.lower.x;
    lower.y = aabb.lower.y;
    lower.z = aabb.lower.z;
    upper.x = aabb.upper.x;
    upper.y = aabb.upper.y;
    upper.z = aabb.upper.z;

    return Mbr<COORD_T, 3>(lower, upper);
  }
};

template <typename COORD_T>
struct PQElement {
  int node;
  COORD_T dist;
  PQElement() : node(-1), dist(0) {}
  PQElement(int node, COORD_T dist) : node(node), dist(dist) {}
  bool operator<(const PQElement& other) const { return dist < other.dist; }
};

template <typename Real, typename Objects, typename AABBGetter,
          typename MortonCodeCalculator>
Real best_first_hd(
    const ::lbvh::bvh<Real, Objects, AABBGetter, MortonCodeCalculator>& tree_a,
    const ::lbvh::bvh<Real, Objects, AABBGetter, MortonCodeCalculator>& tree_b,
    const thrust::host_vector<Objects>& points_a,
    const thrust::host_vector<Objects>& points_b) {
  using bvh_type = ::lbvh::bvh<Real, Objects, AABBGetter, MortonCodeCalculator>;
  using index_type = typename bvh_type::index_type;
  using vector_type = typename lbvh::vector_of<Real>::type;
  constexpr int N_DIMS = vec_info<Objects>::n_dims;

  assert(tree_a.query_host_enabled() && tree_b.query_host_enabled());
  AABBToMbr<Real, N_DIMS> aabb_to_mbr;
  std::priority_queue<PQElement<Real>> pq;
  auto mbr_b = aabb_to_mbr(tree_b.aabbs_host()[0]);

  pq.push(PQElement<Real>(0, std::numeric_limits<Real>::max()));

  Real cmax = 0;
  int visited_nodes = 0;
  int visited_leaves = 0;

  while (!pq.empty()) {
    PQElement<Real> e = pq.top();
    index_type curr_node_idx = e.node;

    pq.pop();
    if (tree_a.nodes_host()[curr_node_idx].object_idx != 0xFFFFFFFF) {
      LOG(INFO) << "Visited Nodes " << visited_nodes << " Visited Leaves "
                << visited_leaves;
      return e.dist;
    }

    const index_type L_idx = tree_a.nodes_host()[curr_node_idx].left_idx;
    const index_type R_idx = tree_a.nodes_host()[curr_node_idx].right_idx;
    auto handle_child = [&](const index_type node_idx) {
      const auto obj_idx = tree_a.nodes_host()[node_idx].object_idx;

      if (obj_idx != 0xFFFFFFFF) {  // leaf node
        auto& obj = tree_a.objects_host()[obj_idx];
        auto nn =
            lbvh::query_host(tree_b, lbvh::nearest(obj),
                             [](const vector_type& a, const Objects& b) {
                               return EuclideanDistance2(
                                   *reinterpret_cast<const Objects*>(&a.x), b);
                             });
        auto dist2 = nn.second;
        CHECK_LE(dist2, e.dist);

        if (dist2 > cmax) {
          pq.push(PQElement<Real>(node_idx, dist2));
          cmax = dist2;
        }
        visited_leaves++;
      } else {  // internal node
        auto mbr = aabb_to_mbr(tree_a.aabbs_host()[node_idx]);
        HdBounds<Real, N_DIMS> bounds(mbr);
        auto dist2 = bounds.GetUpperBound2(mbr_b);

        // if (dist2 > e.dist) {
        //   auto parent_mbr = aabb_to_mbr(tree_a.aabbs_host()[curr_node_idx]);
        //   HdBounds<Real, N_DIMS> parent_bounds(parent_mbr);
        //   auto parent_dist2 = parent_bounds.GetUpperBound2(mbr_b);
        //   printf(
        //       "Parent x [%.8f, %.8f], y [%.8f, %.8f], Child x [%.8f, %.8f], y
        //       "
        //       "[%.8f, %.8f], MbrB x [%.8f, %.8f], y [%.8f, %.8f]\n",
        //       parent_mbr.lower(0), parent_mbr.upper(0), parent_mbr.lower(1),
        //       parent_mbr.upper(1), mbr.lower(0), mbr.upper(0), mbr.lower(1),
        //       mbr.upper(1), mbr_b.lower(0), mbr_b.upper(0), mbr_b.lower(1),
        //       mbr_b.upper(1));
        // }
        // A containing B does not mean HD A > HD B
        // CHECK_LE(dist2, e.dist);
        visited_nodes++;
        if (dist2 > cmax) {
          pq.push(PQElement<Real>(node_idx, dist2));
        }
      }
    };

    if (L_idx < tree_a.nodes_host().size()) {
      handle_child(L_idx);
    }
    if (R_idx < tree_a.nodes_host().size()) {
      handle_child(R_idx);
    }
  }
  // Should not be here
  return -1;
}

}  // namespace detail

template <typename COORD_T, int N_DIMS>
class HausdorffDistanceLBVH {
  using coord_t = COORD_T;
  using point_t = typename cuda_vec<COORD_T, N_DIMS>::type;
  using aabb_getter_t = detail::aabb_getter<coord_t, N_DIMS>;

 public:
  HausdorffDistanceLBVH() = default;

  template <typename IT>
  void SetPointsTo(const Stream& stream, IT begin, IT end) {
    points_b_.assign(begin, end);

    thrust::shuffle(thrust::cuda::par.on(stream.cuda_stream()),
                    points_b_.begin(), points_b_.end(), g_);

    bvh_b_ = lbvh::bvh<COORD_T, point_t, aabb_getter_t>(points_b_.begin(),
                                                        points_b_.end(), true);
  }

  template <typename IT>
  COORD_T CalculateDistanceFrom(const Stream& stream, IT begin, IT end) {
    points_a_.assign(begin, end);

    thrust::shuffle(thrust::cuda::par.on(stream.cuda_stream()),
                    points_a_.begin(), points_a_.end(), g_);

    bvh_a_ = lbvh::bvh<COORD_T, point_t, aabb_getter_t>(points_a_.begin(),
                                                        points_a_.end(), true);

    thrust::host_vector<point_t> points_a(points_a_);
    thrust::host_vector<point_t> points_b(points_b_);

    return sqrt(detail::best_first_hd(bvh_a_, bvh_b_, points_a, points_b));
  }

 private:
  thrust::default_random_engine g_;
  thrust::device_vector<point_t> points_a_;
  thrust::device_vector<point_t> points_b_;
  SharedValue<COORD_T> cmax2_;
  lbvh::bvh<COORD_T, point_t, aabb_getter_t> bvh_a_;
  lbvh::bvh<COORD_T, point_t, aabb_getter_t> bvh_b_;
};
}  // namespace hd

#endif  // HAUSDORFF_DISTANCE_LBVH_H
