#ifndef KD_TREE_H
#define KD_TREE_H
#include <thrust/device_vector.h>

#include <vector>

#include "cukd/builder.h"
#include "cukd/fcp.h"
#include "geoms/mbr.h"
#include "utils/stream.h"
#include "utils/type_traits.h"
namespace hd {

template <typename COORD_T, int N_DIMS>
struct KDTree {
  using point_t = typename cuda_vec<COORD_T, N_DIMS>::type;
  using mbr_t = Mbr<COORD_T, N_DIMS>;

  struct Node {
    /*! split position - which coordinate the plane is at in chosen dim */
    COORD_T pos;

    /*! ID of first child node (if inner node), or offset into
      primIDs[] array, if leaf */
    uint32_t offset;

    /*! number of prims in the leaf (if > 0) or 0 (if inner node) */
    uint16_t count;

    /*! split dimension - which dimension the plane is
      subdividing, if inner node */
    int16_t dim;
  };

  int fcp(const point_t& queryPoint) {
    /*! helper struct to hold the current-best results of a fcp kernel during
     * traversal */
    struct FCPResult {
      COORD_T initialCullDist2() const { return closestDist2; }

      COORD_T clear(COORD_T initialDist2) {
        closestDist2 = initialDist2;
        closestPrimID = -1;
        return closestDist2;
      }

      /*! process a new candidate with given ID and (square) distance;
        and return square distance to be used for subsequent
        queries */
      COORD_T processCandidate(int candPrimID, COORD_T candDist2) {
        if (candDist2 < closestDist2) {
          closestDist2 = candDist2;
          closestPrimID = candPrimID;
        }
        return closestDist2;
      }

      int returnValue() const { return closestPrimID; }

      int closestPrimID;
      COORD_T closestDist2;
    };

    FCPResult result;
    result.clear(INFINITY);

    auto cullDist = result.initialCullDist2();

    /* can do at most 2**30 points... */
    struct StackEntry {
      int nodeID;
      COORD_T sqrDist;
    };
    enum { stack_depth = 50 };
    StackEntry stackBase[stack_depth];
    StackEntry* stackPtr = stackBase;

    /*! current node in the tree we're traversing */
    int nodeID = 0;
    Node node;
    int numSteps = 0;

    while (true) {
      while (true) {
        node = nodes[nodeID];
        ++numSteps;
        if (node.count)
          // this is a leaf...
          break;
        const auto query_coord =
            reinterpret_cast<const COORD_T*>(&queryPoint.x)[node.dim];
        const bool leftIsClose = query_coord < node.pos;
        const int lChild = node.offset + 0;
        const int rChild = node.offset + 1;

        const int closeChild = leftIsClose ? lChild : rChild;
        const int farChild = leftIsClose ? rChild : lChild;

        const float sqrDistToPlane =
            (query_coord - node.pos) * (query_coord - node.pos);
        if (sqrDistToPlane < cullDist) {
          stackPtr->nodeID = farChild;
          stackPtr->sqrDist = sqrDistToPlane;
          ++stackPtr;
          if ((stackPtr - stackBase) >= stack_depth) {
            printf("STACK OVERFLOW %i\n", int(stackPtr - stackBase));
            return -1;
          }
        }
        nodeID = closeChild;
      }

      for (int i = 0; i < node.count; i++) {
        int primID = primIDs[node.offset + i];
        const auto sqrDist = EuclideanDistance2(data[primID], queryPoint);
        cullDist = result.processCandidate(primID, sqrDist);
      }

      while (true) {
        if (stackPtr == stackBase) {
          return result.returnValue();
        }
        --stackPtr;
        if (stackPtr->sqrDist >= cullDist)
          continue;
        nodeID = stackPtr->nodeID;
        break;
      }
    }
  }

  mbr_t bounds;
  std::vector<Node> nodes;
  std::vector<uint32_t> primIDs;
  std::vector<point_t> data;
  int numPrims;
  int numNodes;

  void Build(const Stream& stream, thrust::device_vector<point_t>& points) {
    using data_traits = cukd::default_data_traits<point_t>;
    using gpu_kdtree_t = cukd::SpatialKDTree<point_t, data_traits>;
    gpu_kdtree_t kd_tree;
    cukd::BuildConfig buildConfig{};

    cukd::buildTree(kd_tree, thrust::raw_pointer_cast(points.data()),
                    points.size(), buildConfig, stream.cuda_stream());
    stream.Sync();
    // copy back

    bounds = mbr_t(kd_tree.bounds.lower, kd_tree.bounds.upper);
    nodes.resize(kd_tree.numNodes);
    primIDs.resize(kd_tree.numPrims);
    data.resize(kd_tree.numPrims);
    numPrims = kd_tree.numPrims;
    numNodes = kd_tree.numNodes;

    CUDA_CHECK(cudaMemcpyAsync(nodes.data(), kd_tree.nodes,
                               sizeof(Node) * numNodes, cudaMemcpyDeviceToHost,
                               stream.cuda_stream()));
    CUDA_CHECK(cudaMemcpyAsync(primIDs.data(), kd_tree.primIDs,
                               sizeof(uint32_t) * numPrims,
                               cudaMemcpyDeviceToHost, stream.cuda_stream()));
    CUDA_CHECK(cudaMemcpyAsync(data.data(), kd_tree.data,
                               sizeof(point_t) * numPrims,
                               cudaMemcpyDeviceToHost, stream.cuda_stream()));
  }
};

}  // namespace hd
#endif  // KD_TREE_H
