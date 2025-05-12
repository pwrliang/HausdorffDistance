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
#include "geoms/distance.h"
#include "geoms/hd_bounds.h"
#include "geoms/mbr.h"
#include "hausdorff_distance.h"
#include "index/kd_tree.h"
#include "utils/derived_atomic_functions.h"
#include "utils/helpers.h"
#include "utils/queue.h"
#include "utils/stopwatch.h"
#include "utils/stream.h"
#include "utils/type_traits.h"
#include "utils/util.h"

namespace hd {

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
  bool operator<(const BFSEntry& other) const { return dist < other.dist; }
};
}  // namespace details

template <typename COORD_T, int N_DIMS>
class HausdorffDistanceBranchNBound
    : public HausdorffDistance<COORD_T, N_DIMS> {
  using coord_t = COORD_T;
  using point_t = typename cuda_vec<COORD_T, N_DIMS>::type;
  using data_traits = cukd::default_data_traits<point_t>;
  using mbr_t = Mbr<COORD_T, N_DIMS>;
  using bfs_entry_t = details::BFSEntry<COORD_T, N_DIMS>;

 public:
  HausdorffDistanceBranchNBound() = default;

  COORD_T CalculateDistance(std::vector<point_t>& points_a,
                            std::vector<point_t>& points_b) override {
    double build_time, compute_time;
    Stopwatch sw;
    Stream stream;
    thrust::device_vector<point_t> temp_points;
    sw.start();
    temp_points = points_a;
    kd_tree_a_.Build(stream, temp_points);
    temp_points = points_b;
    kd_tree_b_.Build(stream, temp_points);
    sw.stop();

    build_time = sw.ms();

    sw.start();
    std::priority_queue<bfs_entry_t> pq;
    auto mbr_b = kd_tree_b_.bounds;
    COORD_T cmax2 = 0;
    uint64_t n_compared_pairs = 0;

    pq.push({0, std::numeric_limits<COORD_T>::max(), kd_tree_a_.bounds});

    // Best-first Search (BFS)
    while (!pq.empty()) {
      auto entry = pq.top();
      pq.pop();

      auto node_id = entry.node_id;
      auto dist = entry.dist;
      const auto& mbr = entry.mbr;
      auto n_prims = kd_tree_a_.nodes[node_id].count;
      const auto& node = kd_tree_a_.nodes[node_id];

      if (n_prims == 0) {
        assert(mbr.IsValid());

        // parent MBR is split along node.dim
        mbr_t mbrs[2] = {mbr, mbr};

        mbrs[0].set_upper(node.dim, node.pos);
        mbrs[1].set_lower(node.dim, node.pos);

        assert(mbrs[0].IsValid());
        assert(mbrs[1].IsValid());

        for (int i = 0; i < 2; i++) {
          const auto& child_mbr = mbrs[i];
          HdBounds<COORD_T, N_DIMS> bounds(child_mbr);
          auto upper2 = bounds.GetUpperBound2(mbr_b);
          if (upper2 > cmax2) {
            pq.push(bfs_entry_t(node.offset + i, upper2, child_mbr));
          }
          assert(child_mbr.IsValid());
        }
      } else {
        COORD_T max2 = 0;

        for (uint16_t i = 0; i < n_prims; i++) {
          auto prim_id = kd_tree_a_.primIDs[node.offset + i];
          auto& p = points_a[prim_id];
          auto point_b_id = kd_tree_b_.fcp(p);
          auto dist2 = EuclideanDistance2(p, points_b[point_b_id]);
          n_compared_pairs++;
          max2 = std::max(max2, dist2);
        }
        cmax2 = std::max(cmax2, max2);
      }
    }
    sw.stop();

    compute_time = sw.ms();

    auto& stats = this->stats_;

    stats["Algorithm"] = "Branch-and-Bound";
    stats["Execution"] = "CPU";
    stats["ComparedPairs"] = n_compared_pairs;
    stats["BuildIndexTime"] = build_time;
    stats["ComputeTime"] = compute_time;
    stats["ReportedTime"] = compute_time;

    return sqrt(cmax2);
  }

 private:
  SharedValue<COORD_T> cmax2_;
  KDTree<COORD_T, N_DIMS> kd_tree_a_, kd_tree_b_;
};
}  // namespace hd

#endif  // HAUSDORFF_DISTANCE_BRANCH_N_BOUND_H
