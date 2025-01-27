
#ifndef MORTON_CODE_H
#define MORTON_CODE_H
#include "utils/type_traits.h"
#include "utils/util.h"

namespace hd {
namespace detail {

DEV_HOST_INLINE
std::uint32_t expand_bits(std::uint32_t n) {
  // v = (v * 0x00010001u) & 0xFF0000FFu;
  // v = (v * 0x00000101u) & 0x0F00F00Fu;
  // v = (v * 0x00000011u) & 0xC30C30C3u;
  // v = (v * 0x00000005u) & 0x49249249u;
  // return v;
  n &= 0x0000ffff;
  n = (n | (n << 8)) & 0x00FF00FF;
  n = (n | (n << 4)) & 0x0F0F0F0F;
  n = (n | (n << 2)) & 0x33333333;
  n = (n | (n << 1)) & 0x55555555;
  return n;
}

DEV_HOST_INLINE
uint32_t morton_code(float2 xy, float resolution = 1024.0f) {
  xy.x = ::fminf(::fmaxf(xy.x * resolution, 0.0f), resolution - 1.0f);
  xy.y = ::fminf(::fmaxf(xy.y * resolution, 0.0f), resolution - 1.0f);
  auto xx = expand_bits(static_cast<uint32_t>(xy.x));
  auto yy = expand_bits(static_cast<uint32_t>(xy.y));
  return xx * 2 + yy;
}

DEV_HOST_INLINE
uint32_t morton_code(double2 xy, float resolution = 1024.0f) {
  xy.x = ::fmin(::fmax(xy.x * resolution, 0.0f), resolution - 1.0f);
  xy.y = ::fmin(::fmax(xy.y * resolution, 0.0f), resolution - 1.0f);
  auto xx = expand_bits(static_cast<uint32_t>(xy.x));
  auto yy = expand_bits(static_cast<uint32_t>(xy.y));
  return xx * 2 + yy;
}

DEV_HOST_INLINE
uint32_t morton_code(float3 xyz, float resolution = 1024.0f) {
  xyz.x = ::fminf(::fmaxf(xyz.x * resolution, 0.0f), resolution - 1.0f);
  xyz.y = ::fminf(::fmaxf(xyz.y * resolution, 0.0f), resolution - 1.0f);
  xyz.z = ::fminf(::fmaxf(xyz.z * resolution, 0.0f), resolution - 1.0f);
  auto xx = expand_bits(static_cast<uint32_t>(xyz.x));
  auto yy = expand_bits(static_cast<uint32_t>(xyz.y));
  auto zz = expand_bits(static_cast<uint32_t>(xyz.z));
  return xx * 4 + yy * 2 + zz;
}

DEV_HOST_INLINE
uint32_t morton_code(double3 xyz, float resolution = 1024.0f) {
  xyz.x = ::fmin(::fmax(xyz.x * resolution, 0.0f), resolution - 1.0f);
  xyz.y = ::fmin(::fmax(xyz.y * resolution, 0.0f), resolution - 1.0f);
  xyz.z = ::fmin(::fmax(xyz.z * resolution, 0.0f), resolution - 1.0f);
  auto xx = expand_bits(static_cast<uint32_t>(xyz.x));
  auto yy = expand_bits(static_cast<uint32_t>(xyz.y));
  auto zz = expand_bits(static_cast<uint32_t>(xyz.z));
  return xx * 4 + yy * 2 + zz;
}

}  // namespace detail
}  // namespace hd
#endif  // MORTON_CODE_H
