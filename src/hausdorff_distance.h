#ifndef HAUSDORFF_DISTANCE_H
#define HAUSDORFF_DISTANCE_H
#include <glog/logging.h>
#include <thrust/device_vector.h>

#include <nlohmann/json.hpp>
#include <vector>

#include "utils/stream.h"
#include "utils/type_traits.h"

namespace hd {
template <typename COORD_T, int N_DIMS>
class HausdorffDistance {
 public:
  using point_t = typename cuda_vec<COORD_T, N_DIMS>::type;

  virtual ~HausdorffDistance() {}

  virtual COORD_T CalculateDistance(std::vector<point_t>& points_a,
                                    std::vector<point_t>& points_b) {
    LOG(FATAL) << "CalculateDistance on the CPU is not implemented";
    return -1;
  }

  virtual COORD_T CalculateDistance(const Stream& stream,
                                    thrust::device_vector<point_t>& points_a,
                                    thrust::device_vector<point_t>& points_b) {
    LOG(FATAL) << "CalculateDistance on the GPU is not implemented";
    return -1;
  }

  const nlohmann::json& get_stats() const {
    CHECK(stats_.contains("Algorithm"));
    CHECK(stats_.contains("Execution"));
    CHECK(stats_.contains("ReportedTime"));
    return stats_;
  }

 protected:
  nlohmann::json stats_;
};
}  // namespace hd

#endif  // HAUSDORFF_DISTANCE_H
