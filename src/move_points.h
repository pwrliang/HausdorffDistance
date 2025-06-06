#ifndef MOVE_POINTS_H
#define MOVE_POINTS_H
#include <vector_types.h>
#include <random>
#include <vector>

#include "utils/util.h"

namespace hd {

template <typename POINT_T>
void MovePoints(const std::vector<POINT_T>& points_a,
                std::vector<POINT_T>& points_b, double move_offset, int seed, int distribution, bool bothSide) {
  using coord_t = typename vec_info<POINT_T>::type;
  int n_dims = vec_info<POINT_T>::n_dims;

  auto calculate_average_point = [n_dims](const std::vector<POINT_T>& points) {
    POINT_T average_point;
    memset(&average_point, 0, sizeof(average_point));

    for (int i = 0; i < points.size(); i++) {
      auto* p_point = reinterpret_cast<const coord_t*>(&points[i]);
      for (int dim = 0; dim < n_dims; dim++) {
        reinterpret_cast<coord_t*>(&average_point)[dim] += p_point[dim];
      }
    }
    return average_point;
  };

  auto avg_point_a = calculate_average_point(points_a);
  auto avg_point_b = calculate_average_point(points_b);
  POINT_T dir;

  static std::mt19937 gen(seed);
  std::bernoulli_distribution signDist(0.5);
  std::normal_distribution<float> distn(move_offset, move_offset/4);
  std::uniform_real_distribution<float> distu(move_offset/4*3, move_offset/4*5);

  for (int dim = 0; dim < n_dims; dim++) {
    reinterpret_cast<coord_t*>(&dir)[dim] =
        (reinterpret_cast<coord_t*>(&avg_point_b)[dim] -
         reinterpret_cast<coord_t*>(&avg_point_a)[dim]);
    if (reinterpret_cast<coord_t*>(&dir)[dim] >= 0) {
      reinterpret_cast<coord_t*>(&dir)[dim] = 1;
    } else {
      reinterpret_cast<coord_t*>(&dir)[dim] = -1;
    }
  }

  for (int i = 0; i < points_b.size(); i++) {
    for (int dim = 0; dim < n_dims; dim++) {
	  double tmp;
	  if(distribution == 0){
	  	//uniform
	    tmp = distu(gen);
	  }else if(distribution == 1){
	  	//normal
		tmp = distn(gen);
	  }
	  if(bothSide){
		tmp *= signDist(gen) ? 1.0 : -1.0;
	  }
      reinterpret_cast<coord_t*>(&points_b[i])[dim] +=
          	reinterpret_cast<coord_t*>(&dir)[dim] * tmp;
    }
  }
}
}  // namespace hd

#endif  // MOVE_POINTS_H
