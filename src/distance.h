#ifndef DISTANCE_H
#define DISTANCE_H
#include <vector_types.h>

#include "utils/util.h"
namespace hd {
DEV_HOST_INLINE float EuclideanDistance2(float2 a, float2 b) {
  return (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y);
}

DEV_HOST_INLINE float EuclideanDistance3(float3 a, float3 b) {
  return (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y) +
         (a.z - b.z) * (a.z - b.z);
}

DEV_HOST_INLINE double EuclideanDistance2(double2 a, double2 b) {
  return (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y);
}

DEV_HOST_INLINE double EuclideanDistance3(double3 a, double3 b) {
  return (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y) +
         (a.z - b.z) * (a.z - b.z);
}

}  // namespace hd
#endif  // DISTANCE_H
