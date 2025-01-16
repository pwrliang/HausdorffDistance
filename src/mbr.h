#ifndef MBR_H
#define MBR_H

#include "utils/derived_atomic_functions.h"
#include "utils/shared_value.h"
#include "utils/type_traits.h"
#include "utils/util.h"

namespace hd {
template <typename COORD_T, int N_DIMS>
class Mbr {
 public:
  using coord_t = COORD_T;
  static constexpr int n_dims = N_DIMS;
  using point_t = typename cuda_vec<COORD_T, N_DIMS>::type;

  Mbr() {
    auto* lower = reinterpret_cast<coord_t*>(&lower_.x);
    auto* upper = reinterpret_cast<coord_t*>(&upper_.x);

    for (int dim = 0; dim < n_dims; dim++) {
      lower[dim] = std::numeric_limits<coord_t>::max();
      upper[dim] = std::numeric_limits<coord_t>::lowest();
    }
  }

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
    return reinterpret_cast<const COORD_T*>(&lower_.x)[dim];
  }

  DEV_HOST_INLINE COORD_T upper(int dim) const {
    assert(dim >= 0 && dim < N_DIMS);
    return reinterpret_cast<const COORD_T*>(&upper_.x)[dim];
  }

  DEV_HOST_INLINE void set_lower(int dim, COORD_T val) {
    assert(dim >= 0 && dim < N_DIMS);
    reinterpret_cast<COORD_T*>(&lower_.x)[dim] = val;
  }

  DEV_HOST_INLINE void set_upper(int dim, COORD_T val) {
    assert(dim >= 0 && dim < N_DIMS);
    reinterpret_cast<COORD_T*>(&upper_.x)[dim] = val;
  }

  DEV_HOST_INLINE bool Contains(const Mbr& mbr) const {
    bool contains = true;
    for (int i = 0; contains && i < N_DIMS; i++) {
      contains &= lower(i) <= mbr.lower(i) && upper(i) >= mbr.upper(i);
    }
    return contains;
  }

  DEV_HOST_INLINE point_t Normalize(const point_t& p) const {
    auto* p_lower = reinterpret_cast<const COORD_T*>(&lower_.x);
    auto* p_upper = reinterpret_cast<const COORD_T*>(&upper_.x);
    auto* p_point = reinterpret_cast<const COORD_T*>(&p.x);
    point_t np;

    for (int dim = 0; dim < N_DIMS; dim++) {
      assert(p_point[dim] >= p_lower[dim] && p_point[dim] <= p_upper[dim]);

      auto val = (p_point[dim] - p_lower[dim]) / (p_upper[dim] - p_lower[dim]);
      reinterpret_cast<coord_t*>(&np.x)[dim] = val;
    }
    return np;
  }

  DEV_HOST_INLINE void Expand(const point_t& p) {
    auto* p_lower = reinterpret_cast<COORD_T*>(&lower_.x);
    auto* p_upper = reinterpret_cast<COORD_T*>(&upper_.x);
    auto* p_point = reinterpret_cast<const COORD_T*>(&p.x);

    for (int dim = 0; dim < N_DIMS; dim++) {
      p_lower[dim] = std::min(p_lower[dim], p_point[dim]);
      p_upper[dim] = std::max(p_upper[dim], p_point[dim]);
    }
  }

  DEV_INLINE void ExpandAtomic(const point_t& p) {
    auto* p_lower = reinterpret_cast<COORD_T*>(&lower_.x);
    auto* p_upper = reinterpret_cast<COORD_T*>(&upper_.x);
    auto* p_point = reinterpret_cast<const COORD_T*>(&p.x);

    for (int dim = 0; dim < N_DIMS; dim++) {
      atomicMin(&p_lower[dim], p_point[dim]);
      atomicMax(&p_upper[dim], p_point[dim]);
    }
  }

  DEV_HOST_INLINE void Expand(const Mbr& mbr) {
    for (int dim = 0; dim < N_DIMS; dim++) {
      set_lower(dim, std::min(lower(dim), mbr.lower(dim)));
      set_upper(dim, std::max(upper(dim), mbr.upper(dim)));
    }
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
