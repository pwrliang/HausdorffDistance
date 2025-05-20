#ifndef DISTANCE_H
#define DISTANCE_H
#include <vector_types.h>

#include "utils/util.h"

namespace hd {
DEV_HOST_INLINE int EuclideanDistance2(const int2& a, const int2& b) {
  return (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y);
}

DEV_HOST_INLINE int EuclideanDistance2(const int3& a, const int3& b) {
  return (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y) +
         (a.z - b.z) * (a.z - b.z);
}

DEV_HOST_INLINE float EuclideanDistance2(const float2& a, const float2& b) {
  return (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y);
}

DEV_HOST_INLINE float EuclideanDistance2(const float3& a, const float3& b) {
  return (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y) +
         (a.z - b.z) * (a.z - b.z);
}

DEV_HOST_INLINE double EuclideanDistance2(const double2& a, const double2& b) {
  return (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y);
}

DEV_HOST_INLINE double EuclideanDistance2(const double3& a, const double3& b) {
  return (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y) +
         (a.z - b.z) * (a.z - b.z);
}

}  // namespace hd
#endif  // DISTANCE_H
