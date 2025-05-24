#include <cuda.h>
#include <cuda_runtime.h>
#include <optix.h>
#include <optix_device.h>

#include "geoms/distance.h"
#include "rt/launch_parameters.h"
#include "utils/derived_atomic_functions.h"
#include "utils/helpers.h"

enum { SURFACE_RAY_TYPE = 0, RAY_TYPE_COUNT };
// FLOAT_TYPE is defined by CMakeLists.txt
extern "C" __constant__
    hd::details::LaunchParamsOverlapUniformGrid<FLOAT_TYPE, 2>
        params;

extern "C" __global__ void __intersection__overlap_uniform_grid_2d() {
  auto point_a_id = optixGetPayload_0();
  auto mbr_id = optixGetPrimitiveIndex();
  const auto& point_a = params.points_a[point_a_id];

  const auto& mbr_b = params.mbrs_b[mbr_id];
  auto radius = params.radius;
  auto min_dist2 = mbr_b.GetMinDist2(point_a);
  auto max_dist2 = mbr_b.GetMaxDist2(point_a);

  // if (min_dist2 > radius * radius) {
  //   return;
  // }
  // if (mbr_b.Contains(point_a)) {
    optixSetPayload_1(1);
    optixReportIntersection(0, 0);  // return implicitly
  // } else {
    // optixSetPayload_1(0);
  // }
  //
}

extern "C" __global__ void __anyhit__overlap_uniform_grid_2d() {
  optixTerminateRay();
}

extern "C" __global__ void __raygen__overlap_uniform_grid_2d() {
  const auto& in_queue = params.in_queue;
  float tmin = 0;
  float tmax = FLT_MIN;

  for (auto i = optixGetLaunchIndex().x; i < in_queue.size();
       i += optixGetLaunchDimensions().x) {
    unsigned int point_id_a = in_queue[i];
    const auto& point_a = params.points_a[point_id_a];

    float3 origin;
    origin.x = point_a.x;
    origin.y = point_a.y;
    origin.z = 0;
    float3 dir = {0, 0, 1};

    unsigned int hit = 0;

    optixTrace(params.handle, origin, dir, tmin, tmax, 0,
               OptixVisibilityMask(255),
               OPTIX_RAY_FLAG_NONE,  // OPTIX_RAY_FLAG_NONE,
               SURFACE_RAY_TYPE,     // SBT offset
               RAY_TYPE_COUNT,       // SBT stride
               SURFACE_RAY_TYPE,     // missSBTIndex
               point_id_a, hit);

    if (hit) {
      params.hit_queue.Append(point_id_a);
    } else {
      params.miss_queue.Append(point_id_a);
    }
  }
}
