#include <cuda.h>
#include <cuda_runtime.h>
#include <optix.h>
#include <optix_device.h>

#include "rt/launch_parameters.h"
#include "distance.h"
#include "utils/derived_atomic_functions.h"
#include "utils/helpers.h"

enum { SURFACE_RAY_TYPE = 0, RAY_TYPE_COUNT };
// FLOAT_TYPE is defined by CMakeLists.txt
extern "C" __constant__
    hd::details::LaunchParamsCull<FLOAT_TYPE, 2> params;

extern "C" __global__ void __intersection__cull_2d() {
  using point_t = typename decltype(params)::point_t;
  auto point_a_id = optixGetPayload_0();
  auto point_b_id = optixGetPrimitiveIndex();
  const auto& point_a = params.points_a[point_a_id];
  const auto& point_b = params.points_b[point_b_id];

  auto dist2 = hd::EuclideanDistance2(point_a, point_b);

  if (dist2 <= params.radius * params.radius) {
    optixSetPayload_1(1);
    optixReportIntersection(0, 0);
  }
}

extern "C" __global__ void __anyhit__cull_2d() {
  optixTerminateRay();
}

extern "C" __global__ void __raygen__cull_2d() {
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

    if (hit == 0) {
      params.out_queue.Append(point_id_a);
    }
  }
}
