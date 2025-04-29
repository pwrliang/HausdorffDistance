#ifndef PRIMITIVE_UTILS_H
#define PRIMITIVE_UTILS_H
#include <optix.h>
#include <thrust/transform_reduce.h>

#include <cstdint>

#include "utils/helpers.h"
#include "utils/type_traits.h"
#include "utils/util.h"

#define NEXT_AFTER_ROUNDS (2)

namespace hd {

namespace details {

DEV_HOST_INLINE uint32_t expand_bits(uint32_t v) noexcept {
  v = (v * 0x00010001u) & 0xFF0000FFu;
  v = (v * 0x00000101u) & 0x0F00F00Fu;
  v = (v * 0x00000011u) & 0xC30C30C3u;
  v = (v * 0x00000005u) & 0x49249249u;
  return v;
}

// Calculates a 30-bit Morton code for the
// given 3D point located within the unit cube [0,1].
DEV_HOST_INLINE uint32_t morton_code(float2 xy,
                                     float resolution = 1024.0f) noexcept {
  xy.x = ::fminf(::fmaxf(xy.x * resolution, 0.0f), resolution - 1.0f);
  xy.y = ::fminf(::fmaxf(xy.y * resolution, 0.0f), resolution - 1.0f);
  const uint32_t xx = expand_bits(static_cast<uint32_t>(xy.x));
  const uint32_t yy = expand_bits(static_cast<uint32_t>(xy.y));
  return xx * 2 + yy;
}

DEV_HOST_INLINE uint32_t morton_code(float3 xyz,
                                     float resolution = 1024.0f) noexcept {
  xyz.x = ::fminf(::fmaxf(xyz.x * resolution, 0.0f), resolution - 1.0f);
  xyz.y = ::fminf(::fmaxf(xyz.y * resolution, 0.0f), resolution - 1.0f);
  xyz.z = ::fminf(::fmaxf(xyz.z * resolution, 0.0f), resolution - 1.0f);
  const uint32_t xx = expand_bits(static_cast<uint32_t>(xyz.x));
  const uint32_t yy = expand_bits(static_cast<uint32_t>(xyz.y));
  const uint32_t zz = expand_bits(static_cast<uint32_t>(xyz.z));
  return xx * 4 + yy * 2 + zz;
}

DEV_HOST_INLINE uint32_t morton_code(double2 xy,
                                     double resolution = 1024.0) noexcept {
  xy.x = ::fmin(::fmax(xy.x * resolution, 0.0), resolution - 1.0);
  xy.y = ::fmin(::fmax(xy.y * resolution, 0.0), resolution - 1.0);
  const uint32_t xx = expand_bits(static_cast<uint32_t>(xy.x));
  const uint32_t yy = expand_bits(static_cast<uint32_t>(xy.y));
  return xx * 2 + yy;
}

DEV_HOST_INLINE uint32_t morton_code(double3 xyz,
                                     double resolution = 1024.0) noexcept {
  xyz.x = ::fmin(::fmax(xyz.x * resolution, 0.0), resolution - 1.0);
  xyz.y = ::fmin(::fmax(xyz.y * resolution, 0.0), resolution - 1.0);
  xyz.z = ::fmin(::fmax(xyz.z * resolution, 0.0), resolution - 1.0);
  const uint32_t xx = expand_bits(static_cast<uint32_t>(xyz.x));
  const uint32_t yy = expand_bits(static_cast<uint32_t>(xyz.y));
  const uint32_t zz = expand_bits(static_cast<uint32_t>(xyz.z));
  return xx * 4 + yy * 2 + zz;
}

DEV_HOST_INLINE OptixAabb GetOptixAABB(float2 p, float radius) {
  OptixAabb aabb;
  aabb.minX = p.x - radius;
  aabb.maxX = p.x + radius;
  aabb.minY = p.y - radius;
  aabb.maxY = p.y + radius;
  aabb.minZ = aabb.maxZ = 0;
  return aabb;
}

DEV_HOST_INLINE OptixAabb GetOptixAABB(float3 p, float radius) {
  OptixAabb aabb;
  aabb.minX = p.x - radius;
  aabb.maxX = p.x + radius;
  aabb.minY = p.y - radius;
  aabb.maxY = p.y + radius;
  aabb.minZ = p.z - radius;
  aabb.maxZ = p.z + radius;
  return aabb;
}

DEV_HOST_INLINE OptixAabb GetOptixAABB(int2 p, float radius) {
  OptixAabb aabb;
  aabb.minX = p.x - radius;
  aabb.maxX = p.x + radius;
  aabb.minY = p.y - radius;
  aabb.maxY = p.y + radius;
  aabb.minZ = aabb.maxZ = 0;
  return aabb;
}

DEV_HOST_INLINE OptixAabb GetOptixAABB(int3 p, float radius) {
  OptixAabb aabb;
  aabb.minX = p.x - radius;
  aabb.maxX = p.x + radius;
  aabb.minY = p.y - radius;
  aabb.maxY = p.y + radius;
  aabb.minZ = p.z - radius;
  aabb.maxZ = p.z + radius;
  return aabb;
}

DEV_HOST_INLINE OptixAabb GetOptixAABB(double2 p, double radius) {
  OptixAabb aabb;
  aabb.minX = next_float_from_double(p.x - radius, -1, 2);
  aabb.maxX = next_float_from_double(p.x + radius, 1, 2);
  aabb.minY = next_float_from_double(p.y - radius, -1, 2);
  aabb.maxY = next_float_from_double(p.y + radius, 1, 2);
  aabb.minZ = aabb.maxZ = 0;
  return aabb;
}

DEV_HOST_INLINE OptixAabb GetOptixAABB(double3 p, double radius) {
  OptixAabb aabb;
  aabb.minX = next_float_from_double(p.x - radius, -1, 2);
  aabb.maxX = next_float_from_double(p.x + radius, 1, 2);
  aabb.minY = next_float_from_double(p.y - radius, -1, 2);
  aabb.maxY = next_float_from_double(p.y + radius, 1, 2);
  aabb.minZ = next_float_from_double(p.z - radius, -1, 2);
  aabb.maxZ = next_float_from_double(p.z + radius, 1, 2);
  return aabb;
}

// FIXME: May have precision issue for double type
template <typename COORD_T, int N_DIMS>
DEV_HOST_INLINE OptixAabb GetOptixAABB(const Mbr<COORD_T, N_DIMS>& mbr,
                                       COORD_T radius) {
  OptixAabb aabb;

  aabb.minZ = aabb.maxZ = 0;

  for (int dim = 0; dim < N_DIMS; dim++) {
    auto lower = mbr.lower(dim);
    auto upper = mbr.upper(dim);
    reinterpret_cast<float*>(&aabb.minX)[dim] = lower - radius;
    reinterpret_cast<float*>(&aabb.maxX)[dim] = upper + radius;
  }
  return aabb;
}
}  // namespace details
template <typename POINT_T>
POINT_T CalculateCenterPoint(const Stream& stream,
                             const thrust::device_vector<POINT_T>& points) {
  using coord_t = typename vec_info<POINT_T>::type;
  constexpr int n_dims = vec_info<POINT_T>::n_dims;
  POINT_T center_point;

  for (int dim = 0; dim < n_dims; dim++) {
    auto sum = thrust::transform_reduce(
        thrust::cuda::par.on(stream.cuda_stream()), points.begin(),
        points.end(),
        [=] __device__(const POINT_T& p) {
          return reinterpret_cast<const coord_t*>(&p.x)[dim];
        },
        (coord_t) 0, thrust::plus<coord_t>());
    reinterpret_cast<coord_t*>(&center_point.x)[dim] = sum / points.size();
  }

  return center_point;
}
}  // namespace hd
#endif  // PRIMITIVE_UTILS_H
