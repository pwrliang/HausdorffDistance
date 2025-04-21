#ifndef HAUSDORFF_DISTANCE_BRANCH_N_BOUND_H
#define HAUSDORFF_DISTANCE_BRANCH_N_BOUND_H
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
#include "utils/queue.h"
#include "utils/stopwatch.h"
#include "utils/stream.h"
#include "utils/type_traits.h"
#include "utils/util.h"

namespace hd {

namespace dev {

namespace details {
template <typename COORD_T, int N_DIMS>
struct BFSEntry {
  using mbr_t = Mbr<COORD_T, N_DIMS>;
  uint32_t node_id;
  COORD_T dist;
  mbr_t mbr;
  bool leaf;
  BFSEntry() = default;
  DEV_HOST BFSEntry(uint32_t _node_id, COORD_T _dist, const mbr_t& _mbr)
      : node_id(_node_id), dist(_dist), mbr(_mbr), leaf(false) {}

  DEV_HOST explicit BFSEntry(COORD_T _dist) : dist(_dist), leaf(true) {}
};
}  // namespace details

}  // namespace dev

template <typename COORD_T, int N_DIMS>
class HausdorffDistanceBranchNBound {
  using coord_t = COORD_T;
  using point_t = typename cuda_vec<COORD_T, N_DIMS>::type;
  using data_traits = cukd::default_data_traits<point_t>;
  using mbr_t = Mbr<COORD_T, N_DIMS>;

 public:
  HausdorffDistanceBranchNBound() = default;

  COORD_T CalculateDistance(const Stream& stream,
                            thrust::device_vector<point_t>& points_a,
                            thrust::device_vector<point_t>& points_b) {
    using kdtree_t = cukd::SpatialKDTree<point_t, data_traits>;
    using bfs_entry_t = dev::details::BFSEntry<COORD_T, N_DIMS>;
    kdtree_t tree_a, tree_b;
    cukd::BuildConfig buildConfig{};
    ArrayView<point_t> v_points_a(points_a);
    ArrayView<point_t> v_points_b(points_b);

    Stopwatch sw;

    sw.start();
    cukd::buildTree(tree_a, v_points_a.data(), v_points_a.size(), buildConfig,
                    stream.cuda_stream());
    cukd::buildTree(tree_b, v_points_b.data(), v_points_b.size(), buildConfig,
                    stream.cuda_stream());
    stream.Sync();
    sw.stop();
    LOG(INFO) << "Build Tree Time " << sw.ms();

    mbr_t mbr_a(tree_a.bounds.lower, tree_a.bounds.upper),
        mbr_b(tree_b.bounds.lower, tree_b.bounds.upper);

    Queue<bfs_entry_t> in_queue, out_queue;

    in_queue.Init(points_a.size());
    out_queue.Init(points_a.size());

    cmax2_.set(stream.cuda_stream(), 0);
    in_queue.Clear(stream.cuda_stream());
    out_queue.Clear(stream.cuda_stream());

    in_queue.Append(stream.cuda_stream(),
                    bfs_entry_t(0, std::numeric_limits<COORD_T>::max(), mbr_a));
    size_t in_size = 1;
    auto* p_cmax2 = cmax2_.data();

    while (in_size > 0) {
      auto d_out_queue = out_queue.DeviceObject();

      thrust::for_each(thrust::cuda::par.on(stream.cuda_stream()),
                       in_queue.data(), in_queue.data() + in_size,
                       [=] __device__(const bfs_entry_t& entry) mutable {
                         auto node_id = entry.node_id;
                         auto dist = entry.dist;
                         const auto& mbr = entry.mbr;
                         auto n_prims = tree_a.nodes[node_id].count;
                         const auto& node = tree_a.nodes[node_id];

                         if (n_prims == 0) {
                           assert(mbr.IsValid());

                           mbr_t mbrs[2] = {mbr, mbr};

                           mbrs[0].set_upper(node.dim, node.pos);
                           mbrs[1].set_lower(node.dim, node.pos);

                           assert(mbrs[0].IsValid());
                           assert(mbrs[1].IsValid());

                           for (int i = 0; i < 2; i++) {
                             const auto& child_mbr = mbrs[i];
                             HdBounds<COORD_T, N_DIMS> bounds(child_mbr);
                             auto upper2 = bounds.GetUpperBound2(mbr_b);
                             if (upper2 > *p_cmax2) {
                               d_out_queue.Append(bfs_entry_t(
                                   node.offset + i, upper2, child_mbr));
                             }
                             assert(child_mbr.IsValid());
                           }
                         } else {
                           COORD_T max2 = 0;

                           for (uint16_t i = 0; i < n_prims; i++) {
                             auto prim_id = tree_a.primIDs[node.offset + i];
                             auto& p = v_points_a[prim_id];
                             auto point_b_id = cukd::cct::fcp(tree_b, p);
                             auto dist2 =
                                 EuclideanDistance2(p, v_points_b[point_b_id]);

                             max2 = std::max(max2, dist2);
                           }
                           atomicMax(p_cmax2, max2);
                         }
                       });
      auto out_size = out_queue.size(stream.cuda_stream());

      in_queue.Clear(stream.cuda_stream());
      in_queue.Swap(out_queue);
      in_size = out_size;
    }
    return sqrt(cmax2_.get(stream.cuda_stream()));
  }

 private:
  SharedValue<COORD_T> cmax2_;
};
}  // namespace hd

#endif  // HAUSDORFF_DISTANCE_BRANCH_N_BOUND_H
