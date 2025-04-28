#ifndef MBR_H
#define MBR_H

#include "geoms/distance.h"
#include "utils/derived_atomic_functions.h"
#include "utils/shared_value.h"
#include "utils/stream.h"
#include "utils/type_traits.h"
#include "utils/util.h"

namespace hd {

namespace details {

template <typename COORD_T, int N_DIMS>
struct CornerPointsGetter {
  DEV_HOST_INLINE void operator()(
      const typename cuda_vec<COORD_T, N_DIMS>::type& lower,
      const typename cuda_vec<COORD_T, N_DIMS>::type& upper,
      typename cuda_vec<COORD_T, N_DIMS>::type* corners) {}
};

template <typename COORD_T>
struct CornerPointsGetter<COORD_T, 2> {
  using point_t = typename cuda_vec<COORD_T, 2>::type;
  static constexpr int N_CORNERS = 4;
  DEV_HOST_INLINE void operator()(const point_t& lower, const point_t& upper,
                                  point_t* corners) {
    corners[0] = point_t{lower.x, lower.y};
    corners[1] = point_t{lower.x, upper.y};
    corners[2] = point_t{upper.x, lower.y};
    corners[3] = point_t{upper.x, upper.y};
  }
};

template <typename COORD_T>
struct CornerPointsGetter<COORD_T, 3> {
  using point_t = typename cuda_vec<COORD_T, 3>::type;
  static constexpr int N_CORNERS = 8;
  DEV_HOST_INLINE void operator()(const point_t& lower, const point_t& upper,
                                  point_t* corners) {
    corners[0] = point_t{lower.x, lower.y, lower.z};
    corners[1] = point_t{lower.x, lower.y, upper.z};
    corners[2] = point_t{lower.x, upper.y, lower.z};
    corners[3] = point_t{lower.x, upper.y, upper.z};
    corners[4] = point_t{upper.x, lower.y, lower.z};
    corners[5] = point_t{upper.x, lower.y, upper.z};
    corners[6] = point_t{upper.x, upper.y, lower.z};
    corners[7] = point_t{upper.x, upper.y, upper.z};
  }
};
}  // namespace details

template <typename COORD_T, int N_DIMS>
class Mbr {
 public:
  using coord_t = COORD_T;
  static constexpr int n_dims = N_DIMS;
  using point_t = typename cuda_vec<COORD_T, N_DIMS>::type;

  DEV_HOST Mbr() {
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

  DEV_HOST_INLINE COORD_T get_extent(int dim) const {
    return upper(dim) - lower(dim);
  }

  DEV_HOST_INLINE COORD_T get_volume() const {
    COORD_T measure = 1;
    for (int dim = 0; dim < N_DIMS; dim++) {
      measure *= get_extent(dim);
    }
    return measure;
  }

  DEV_HOST_INLINE void set_lower(int dim, COORD_T val) {
    assert(dim >= 0 && dim < N_DIMS);
    reinterpret_cast<COORD_T*>(&lower_.x)[dim] = val;
  }

  DEV_HOST_INLINE void set_upper(int dim, COORD_T val) {
    assert(dim >= 0 && dim < N_DIMS);
    reinterpret_cast<COORD_T*>(&upper_.x)[dim] = val;
  }

  DEV_HOST_INLINE bool Contains(const point_t& p) const {
    bool contains = true;
    for (int i = 0; contains && i < N_DIMS; i++) {
      auto val = reinterpret_cast<const COORD_T*>(&p.x)[i];
      contains &= lower(i) <= val && upper(i) >= val;
    }
    return contains;
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

  void Expand(const std::vector<point_t>& points) {
    for (auto& p : points) {
      Expand(p);
    }
  }

  point_t TranslateFrom(const point_t& p) {
    assert(Contains(p));
    point_t new_p = p;
    for (int dim = 0; dim < N_DIMS; dim++) {
      reinterpret_cast<COORD_T*>(&new_p.x)[dim] =
          reinterpret_cast<const COORD_T*>(&p.x)[dim] - lower(dim);
    }
    return new_p;
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

  DEV_HOST_INLINE COORD_T GetMaxDist2(const point_t& p) const {
    using getter_t = details::CornerPointsGetter<COORD_T, N_DIMS>;
    getter_t getter;
    point_t corners[getter_t::N_CORNERS];

    getter(lower_, upper_, corners);
    COORD_T max_dist2 = 0;

    for (int i = 0; i < getter_t::N_CORNERS; i++) {
      auto dist2 = EuclideanDistance2(corners[i], p);
      max_dist2 = std::max(max_dist2, dist2);
    }
    return max_dist2;
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

  DEV_HOST_INLINE point_t get_center() const {
    point_t c;

    for (uint32_t i = 0; i < N_DIMS; ++i) {
      auto v_min = lower(i);
      auto v_max = upper(i);
      reinterpret_cast<COORD_T*>(&c.x)[i] = (v_min + v_max) / 2;
    }
    return c;
  }

  DEV_HOST_INLINE bool IsValid() const {
    for (int dim = 0; dim < N_DIMS; ++dim) {
      if (lower(dim) > upper(dim)) {
        return false;
      }
    }
    return true;
  }

 private:
  point_t lower_, upper_;
};

template <typename POINT_T>
using MbrTypeFromPoint =
    Mbr<typename vec_info<POINT_T>::type, vec_info<POINT_T>::n_dims>;

template <typename IT_T>
MbrTypeFromPoint<typename std::iterator_traits<IT_T>::value_type> CalculateMbr(
    const Stream& stream, IT_T begin, IT_T end) {
  using mbr_t =
      MbrTypeFromPoint<typename std::iterator_traits<IT_T>::value_type>;
  using point_t = typename std::iterator_traits<IT_T>::value_type;

  SharedValue<mbr_t> mbr;
  auto* p_mbr = mbr.data();

  mbr.set(stream.cuda_stream(), mbr_t());
  thrust::for_each(
      thrust::cuda::par.on(stream.cuda_stream()), begin, end,
      [=] __device__(const point_t& p) mutable { p_mbr->ExpandAtomic(p); });
  return mbr.get(stream.cuda_stream());
}

}  // namespace hd

#endif  // MBR_H
