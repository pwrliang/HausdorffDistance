#ifndef MBR_H
#define MBR_H

#include "utils/type_traits.h"
#include "utils/util.h"
namespace hd {
template <typename COORD_T, int N_DIMS>
class Mbr {
 public:
  using point_t = typename cuda_vec<COORD_T, N_DIMS>::type;

  Mbr() = default;

  DEV_HOST Mbr(const point_t& lower, const point_t& upper)
      : lower_(lower), upper_(upper) {
    auto* p_lower = reinterpret_cast<const COORD_T*>(&lower.x);
    auto* p_upper = reinterpret_cast<const COORD_T*>(&upper.x);
    for (int i = 0; i < N_DIMS; i++) {
      assert(p_lower[i] <= p_upper[i]);
    }
  }

  DEV_HOST_INLINE const point_t& lower() const { return lower_; }

  DEV_HOST_INLINE const point_t& upper() const { return upper_; }

  DEV_HOST_INLINE COORD_T lower(int dim) const {
    assert(dim >= 0 && dim < N_DIMS);
    return reinterpret_cast<const COORD_T*>(&lower_)[dim];
  }

  DEV_HOST_INLINE COORD_T upper(int dim) const {
    assert(dim >= 0 && dim < N_DIMS);
    return reinterpret_cast<const COORD_T*>(&upper_)[dim];
  }

  DEV_HOST_INLINE bool Contains(const Mbr& mbr) const {
    bool contains = true;
    for (int i = 0; contains && i < N_DIMS; i++) {
      contains &= lower(i) <= mbr.lower(i) && upper(i) >= mbr.upper(i);
    }
    return contains;
  }

  DEV_HOST_INLINE COORD_T GetMinDist2(const Mbr& other) const {
    COORD_T dist2 = 0;

    for (uint32_t i = 0; i < N_DIMS; ++i) {
      auto v_min = lower(i);
      auto v_max = upper(i);
      auto other_v_min = other.lower(i);
      auto other_v_max = other.upper(i);
      COORD_T diff = 0;

      // if this dim has no overlap
      if (other_v_max < v_min) {
        diff = v_min - other_v_max;
      } else if (v_max < other_v_min) {
        diff = other_v_min - v_max;
      }

      dist2 += diff * diff;
    }
    return dist2;
  }

  DEV_HOST_INLINE COORD_T GetMinDist2(const point_t& p) const {
    COORD_T dist2 = 0;

    for (uint32_t i = 0; i < N_DIMS; ++i) {
      auto v_min = lower(i);
      auto v_max = upper(i);
      auto v = reinterpret_cast<const COORD_T*>(&p.x)[i];
      COORD_T diff = 0;

      if (v < v_min) {
        diff = v_min - v;
      } else if (v > v_max) {
        diff = v - v_max;
      }
      dist2 += diff * diff;
    }
    return dist2;
  }

  DEV_HOST_INLINE COORD_T GetMaxDist2(const Mbr& other) const {
    COORD_T dist2 = 0;

    for (uint32_t i = 0; i < N_DIMS; ++i) {
      auto diff1 = std::abs(lower(i) - other.upper(i));
      auto diff2 = std::abs(upper(i) - other.lower(i));
      auto diff = std::max(diff1, diff2);
      /*
      printf(
          "Dim %d, diff %.8f, diff1 %.8f, diff2 %.8f, this [%.8f, %.8f], other "
          "[%.8f, %.8f]\n",
          i, diff, diff1, diff2, lower(i), upper(i), other.lower(i),
          other.upper(i));
      */
      dist2 += diff * diff;
    }
    return dist2;
  }

 private:
  point_t lower_, upper_;
};

}  // namespace hd

#endif  // MBR_H
