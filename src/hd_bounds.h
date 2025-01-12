#ifndef HD_BOUNDS_H
#define HD_BOUNDS_H

#include "distance.h"
#include "mbr.h"
namespace hd {
namespace details {

/**
 * A region defined by end points
 * @tparam COORD_T
 * @tparam N_DIMS
 */
template <typename COORD_T, int N_DIMS>
class Region {};

template <typename COORD_T>
class Region<COORD_T, 2> {
  using point_t = typename cuda_vec<COORD_T, 2>::type;
  using mbr_t = Mbr<COORD_T, 2>;

 public:
  static constexpr int N_FACES = 4;
  static constexpr int N_POINTS = 4;

  Region() = default;

  DEV_HOST explicit Region(const Mbr<COORD_T, 2>& mbr) {
    points_[0] = mbr.lower();
    points_[1].x = mbr.upper(0);
    points_[1].y = mbr.lower(1);
    points_[2] = mbr.upper();
    points_[3].x = mbr.lower(0);
    points_[3].y = mbr.upper(1);
  }

  DEV_HOST_INLINE mbr_t GetFace(int face_id) const {
    mbr_t faces[4] = {
        mbr_t(points_[0], points_[1]),  // bottom
        mbr_t(points_[1], points_[2]),  // right
        mbr_t(points_[3], points_[2]),  // top
        mbr_t(points_[0], points_[3])   // left
    };
    return faces[face_id];
  }

  const point_t* GetPoints() const { return points_; }

 private:
  point_t points_[4];
};

template <typename COORD_T>
class Region<COORD_T, 3> {
  using point_t = typename cuda_vec<COORD_T, 3>::type;
  using mbr_t = Mbr<COORD_T, 3>;

 public:
  static constexpr int N_FACES = 6;
  static constexpr int N_POINTS = 8;

  Region() = default;

  DEV_HOST explicit Region(const Mbr<COORD_T, 3>& mbr) {
    points_[0] = mbr.lower();
    points_[1].x = mbr.upper(0);
    points_[1].y = mbr.lower(1);
    points_[1].z = mbr.lower(2);

    points_[2].x = mbr.lower(0);
    points_[2].y = mbr.upper(1);
    points_[2].z = mbr.lower(2);

    points_[3].x = mbr.upper(0);
    points_[3].y = mbr.upper(1);
    points_[3].z = mbr.lower(2);

    points_[4].x = mbr.lower(0);
    points_[4].y = mbr.lower(1);
    points_[4].z = mbr.upper(2);

    points_[5].x = mbr.upper(0);
    points_[5].y = mbr.lower(1);
    points_[5].z = mbr.upper(2);

    points_[6].x = mbr.lower(0);
    points_[6].y = mbr.upper(1);
    points_[6].z = mbr.upper(2);
    points_[7] = mbr.upper();
  }

  DEV_HOST_INLINE const point_t* GetPoints() const { return points_; }

  DEV_HOST_INLINE mbr_t GetFace(int face_id) {
    mbr_t faces[6] = {
        mbr_t(points_[0], points_[5]),  // bottom
        mbr_t(points_[2], points_[7]),  // top
        mbr_t(points_[0], points_[6]),  // left
        mbr_t(points_[1], points_[7]),  // right
        mbr_t(points_[0], points_[3]),  // front
        mbr_t(points_[4], points_[7])   // back;
    };
    return faces[face_id];
  }

 private:
  point_t points_[8];
};
}  // namespace details

template <typename COORD_T, int N_DIMS>
class HdBounds {
  using mbr_t = Mbr<COORD_T, N_DIMS>;
  using point_t = typename cuda_vec<COORD_T, N_DIMS>::type;
  using region_t = details::Region<COORD_T, N_DIMS>;

 public:
  HdBounds() = default;

  DEV_HOST HdBounds(const mbr_t& mbr) : region_(mbr) {}

  DEV_HOST_INLINE COORD_T GetLowerBound2(const mbr_t& other_mbr) {
    COORD_T dist2 = 0;

    for (int i = 0; i < region_t::N_FACES; ++i) {
      dist2 = std::max(dist2, region_.GetFace(i).GetMinDist2(other_mbr));
    }
    return dist2;
  }

  DEV_HOST_INLINE COORD_T GetLowerBound(const mbr_t& other_mbr) {
    return sqrt(GetLowerBound2(other_mbr));
  }

  DEV_HOST_INLINE COORD_T GetUpperBound2(const mbr_t& other_mbr) {
    COORD_T max_dist2 = 0;
    region_t other_region(other_mbr);

    for (int i = 0; i < region_t::N_FACES; i++) {
      auto face1 = region_.GetFace(i);
      COORD_T min_dist2 = std::numeric_limits<COORD_T>::max();
      for (int j = 0; j < region_t::N_FACES; j++) {
        auto face2 = other_region.GetFace(j);
        auto dist2 = face1.GetMaxDist2(face2);
        if (dist2 <= max_dist2) {
          break;
        }
        min_dist2 = std::min(min_dist2, dist2);
      }
      if (min_dist2 != std::numeric_limits<COORD_T>::max()) {
        max_dist2 = std::max(max_dist2, min_dist2);
      }
    }
    return max_dist2;
  }

  DEV_HOST_INLINE COORD_T GetUpperBound(const mbr_t& other_mbr) {
    return sqrt(GetUpperBound2(other_mbr));
  }

  DEV_HOST_INLINE COORD_T GetLowerBound2(const point_t& other_p) {
    COORD_T dist2 = 0;

    for (int i = 0; i < region_t::N_FACES; ++i) {
      dist2 = std::max(dist2, region_.GetFace(i).GetMinDist2(other_p));
    }
    return dist2;
  }

  DEV_HOST_INLINE COORD_T GetLowerBound(const point_t& other_p) {
    return sqrt(GetLowerBound2(other_p));
  }

  DEV_HOST_INLINE COORD_T GetUpperBound2(const point_t& other_p) {
    auto* p_points = region_.GetPoints();
    COORD_T dist2 = 0;

    for (int i = 0; i < region_t::N_POINTS; i++) {
      const auto& p = p_points[i];
      dist2 = std::max(dist2, EuclideanDistance2(p, other_p));
    }
    return dist2;
  }

  DEV_HOST_INLINE COORD_T GetUpperBound(const point_t& other_p) {
    return sqrt(GetUpperBound2(other_p));
  }

 private:
  region_t region_;
};

}  // namespace hd
#endif  // HD_BOUNDS_H
