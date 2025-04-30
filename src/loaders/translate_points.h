#ifndef HAUSDORFF_DISTANCE_LOADERS_TRANSLATE_POINTS_H
#define HAUSDORFF_DISTANCE_LOADERS_TRANSLATE_POINTS_H
#include <vector_types.h>

#include <vector>

#include "geoms/mbr.h"
#include "utils/type_traits.h"
namespace hd {

template <typename POINT_T>
void MoveToOrigin(std::vector<POINT_T>& points) {
  using coord_t = typename vec_info<POINT_T>::type;
  constexpr int n_dims = vec_info<POINT_T>::n_dims;
  using mbr_t = Mbr<coord_t, n_dims>;
  mbr_t mbr;

  mbr.Expand(points);

  for (auto& p : points) {
    for (int dim = 0; dim < n_dims; dim++) {
      reinterpret_cast<coord_t*>(&p.x)[dim] -= mbr.lower(dim);
    }
  }
}

template <typename POINT_T>
void NormalizePoints(std::vector<POINT_T>& points) {
  using coord_t = typename vec_info<POINT_T>::type;
  constexpr int n_dims = vec_info<POINT_T>::n_dims;
  using mbr_t = Mbr<coord_t, n_dims>;
  mbr_t mbr;

  mbr.Expand(points);
  coord_t max_extent = 0;
  for (int dim = 0; dim < n_dims; dim++) {
    max_extent = std::max(max_extent, mbr.get_extent(dim));
  }
  if (max_extent == 0) {
    max_extent = 1;
  }
  auto invScale = 1.0 / max_extent;
  // assuming the points have moved to origin
  for (auto& p : points) {
    for (int dim = 0; dim < n_dims; dim++) {
      CHECK_GE(reinterpret_cast<coord_t*>(&p.x)[dim], 0);
      reinterpret_cast<coord_t*>(&p.x)[dim] /= max_extent;
      CHECK_LE(reinterpret_cast<coord_t*>(&p.x)[dim], 1);
    }
  }
}

template <typename POINT_T>
void TranslatePoints(std::vector<POINT_T>& points, int translate_dim,
                     double translate_ratio) {
  using coord_t = typename vec_info<POINT_T>::type;
  constexpr int n_dims = vec_info<POINT_T>::n_dims;
  using mbr_t = Mbr<coord_t, n_dims>;
  mbr_t mbr;

  mbr.Expand(points);

  for (int i = 0; i < points.size(); i++) {
    reinterpret_cast<coord_t*>(&points[i].x)[translate_dim] +=
        mbr.get_extent(translate_dim) * translate_ratio;
  }
}
}  // namespace hd

#endif  // HAUSDORFF_DISTANCE_LOADERS_TRANSLATE_POINTS_H
