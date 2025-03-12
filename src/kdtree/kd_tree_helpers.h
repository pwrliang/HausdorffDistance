#pragma once
#include <cooperative_groups.h>

#include "cukd/box.h"
#include "cukd/helpers.h"
#include "kdtree/kd_tree_helpers.h"
#include "utils/array_view.h"
#include "utils/queue.h"

#define WARP_SIZE (32)
namespace cg = cooperative_groups;

DEV_HOST_INLINE float2 operator+(const float2& a, const float2& b) {
  return make_float2(a.x + b.x, a.y + b.y);
}

DEV_HOST_INLINE float2 operator*(const float2& a, const float s) {
  return make_float2(a.x * s, a.y * s);
}

DEV_HOST_INLINE float2 operator/(const float2& a, const float s) {
  float inv = 1.0f / s;
  return a * inv;
}

namespace hd {
template <typename VISITOR_T, int STACK_SIZE = 30>
DEV_INLINE void TraverseRegionSerial(const uint32_t tree_size,
                                     const uint32_t node_id,
                                     VISITOR_T visitor) {
  uint32_t stack_base[STACK_SIZE];
  uint32_t* stack_ptr = stack_base;
  uint32_t curr_node_id = node_id;

  // inorder traverse a binary tree
  while (curr_node_id < tree_size || stack_ptr != stack_base) {
    // move to leftmost node
    while (curr_node_id < tree_size) {
      *stack_ptr++ = curr_node_id;
      curr_node_id = cukd::BinaryTree::leftChildOf(curr_node_id);
    }

    curr_node_id = *(--stack_ptr);

    visitor(curr_node_id);

    curr_node_id = cukd::BinaryTree::rightChildOf(curr_node_id);
  }
  // traverse ancestors
  curr_node_id = node_id;
  auto parent = cukd::BinaryTree::parentOf(curr_node_id);
  auto right = cukd::BinaryTree::rightChildOf(parent);

  // I'm right child or the only child of the parent
  while (parent >= 0 && (curr_node_id == right || right >= tree_size)) {
    visitor(curr_node_id);

    curr_node_id = parent;
    parent = cukd::BinaryTree::parentOf(curr_node_id);
    right = cukd::BinaryTree::rightChildOf(parent);
  }
}

template <typename VISITOR_T, int STACK_SIZE = 30>
DEV_INLINE void InorderTraverseSerial(uint32_t tree_size, uint32_t node_id,
                                      VISITOR_T visitor) {
  uint32_t stack_base[STACK_SIZE];
  uint32_t* stack_ptr = stack_base;

  // inorder traverse a binary tree
  while (node_id < tree_size || stack_ptr != stack_base) {
    // move to leftmost node
    while (node_id < tree_size) {
      *stack_ptr++ = node_id;
      node_id = cukd::BinaryTree::leftChildOf(node_id);
    }

    node_id = *(--stack_ptr);

    visitor(node_id);

    node_id = cukd::BinaryTree::rightChildOf(node_id);
  }
}

template <typename VISITOR_T, int STACK_SIZE = 30>
__global__ void InorderTraverse(uint32_t tree_size,
                                ArrayView<uint32_t> node_ids,
                                VISITOR_T visitor) {
  uint32_t stack_base[STACK_SIZE];

  for (auto i = TID_1D; i < node_ids.size(); i += TOTAL_THREADS_1D) {
    auto node_id = node_ids[i];
    uint32_t* stack_ptr = stack_base;

    // inorder traverse a binary tree
    while (node_id < tree_size || stack_ptr != stack_base) {
      // move to leftmost node
      while (node_id < tree_size) {
        *stack_ptr++ = node_id;
        node_id = cukd::BinaryTree::leftChildOf(node_id);
      }

      node_id = *(--stack_ptr);

      visitor(i, node_id);

      node_id = cukd::BinaryTree::rightChildOf(node_id);
    }
  }
}

template <typename VISITOR_T, int STACK_SIZE = 30>
__global__ void TraverseRegion(uint32_t tree_size, ArrayView<uint32_t> node_ids,
                               VISITOR_T visitor) {
  uint32_t stack_base[STACK_SIZE];

  for (auto i = TID_1D; i < node_ids.size(); i += TOTAL_THREADS_1D) {
    auto node_id = node_ids[i];
    uint32_t* stack_ptr = stack_base;

    // inorder traverse a sub-tree of the node
    while (node_id < tree_size || stack_ptr != stack_base) {
      // move to leftmost node
      while (node_id < tree_size) {
        *stack_ptr++ = node_id;
        node_id = cukd::BinaryTree::leftChildOf(node_id);
      }

      node_id = *(--stack_ptr);

      visitor(i, node_id);

      node_id = cukd::BinaryTree::rightChildOf(node_id);
    }
    // traverse ancestors
    node_id = node_ids[i];
    auto parent = cukd::BinaryTree::parentOf(node_id);
    auto right = cukd::BinaryTree::rightChildOf(parent);

    // I'm right child or the only child of the parent
    while (parent >= 0 && (node_id == right || right >= tree_size)) {
      visitor(i, parent);

      node_id = parent;
      parent = cukd::BinaryTree::parentOf(node_id);
      right = cukd::BinaryTree::rightChildOf(parent);
    }
  }
}

__global__ void CollectRegionsWithNumPoints(
    uint32_t num_nodes, uint32_t max_size, dev::Queue<uint32_t> qualified_nodes,
    dev::Queue<uint32_t> in_q, dev::Queue<uint32_t> out_q) {
  cg::grid_group grid = cg::this_grid();
  auto tree = cukd::ArbitraryBinaryTree(num_nodes);

  if (TID_1D == 0) {
    in_q.Append(0);
  }

  grid.sync();

  while (in_q.size() > 0) {
    for (auto i = TID_1D; i < in_q.size(); i += TOTAL_THREADS_1D) {
      auto node = in_q[i];

      if (tree.numNodesInSubtree(node) > max_size) {
        auto lchild = cukd::BinaryTree::leftChildOf(node);

        if (lchild < num_nodes) {
          out_q.Append(lchild);
          // If left does not exist, right also does not exist
          auto rchild = cukd::BinaryTree::rightChildOf(node);
          if (rchild < num_nodes) {
            out_q.Append(rchild);
          }
        }
      } else {
        qualified_nodes.Append(node);
      }
    }

    grid.sync();
    if (TID_1D == 0) {
      in_q.Clear();
    }
    grid.sync();
    in_q.Swap(out_q);
    grid.sync();
  }
}

template <typename data_t>
__global__ void CalculateBounds(ArrayView<data_t> points,
                                ArrayView<uint32_t> node_ids,
                                ArrayView<cukd::box_t<data_t>> bounds) {
  for (auto i = TID_1D; i < node_ids.size(); i += TOTAL_THREADS_1D) {
    auto current = node_ids[i];
    uint32_t stack_base[30];
    uint32_t* stack_ptr = stack_base;

    cukd::box_t<data_t>& box = bounds[i];

    box.lower = box.upper = points[current];

    while (current < points.size() || stack_ptr != stack_base) {
      // move to leftmost node
      while (current < points.size()) {
        *stack_ptr++ = current;
        current = cukd::BinaryTree::leftChildOf(current);
      }

      current = *(--stack_ptr);

      box.grow(points[current]);

      current = cukd::BinaryTree::rightChildOf(current);
    }
  }
}
}  // namespace hd