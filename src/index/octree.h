#ifndef OCTREE_H
#define OCTREE_H
#include <cstdint>

#include "utils/type_traits.h"
#include "geoms/mbr.h"

namespace hd {

template <typename COORD_T, int N_DIMS>
class Octree {
  using point_t = typename cuda_vec<COORD_T, N_DIMS>::type;
  constexpr static int children_num = 1 << N_DIMS;
  using mbr_t = Mbr<COORD_T, N_DIMS>;

 public:
  struct Node {
    point_t center;
    point_t half_size;
    Node* children[children_num];
    uint32_t point_idx;

    Node() {
      for (int i = 0; i < children_num; ++i) {
        children[i] = nullptr;
      }
    }

    int getOctant(const point_t& p) {
      int octant = 0;
      int bit = 1;
      for (int i = 0; i < N_DIMS; ++i) {
        if (reinterpret_cast<const COORD_T*>(&p.x) >=
            reinterpret_cast<COORD_T*>(&center.x)) {
          octant |= bit;
        }
        bit <<= 1;
      }
      return octant;
    }
  };

  Octree() = default;



 private:
}

}  // namespace hd
#endif  // OCTREE_H
