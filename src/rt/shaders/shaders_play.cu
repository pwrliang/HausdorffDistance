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
    hd::details::LaunchParamsPlay params;

extern "C" __global__ void __intersection__play() {
  auto point_b_id = optixGetPrimitiveIndex();

  printf("Hit %u\n", point_b_id);
}


extern "C" __global__ void __anyhit__play() {
  optixTerminateRay();
}

extern "C" __global__ void __raygen__play() {
  float tmin = 0;
  float tmax = FLT_MIN;

  float3 origin;
  origin.x = 0.5;
  origin.y = 0.5;
  origin.z = 0.5;
  float3 dir = {0, 0, 1};

  unsigned int tmp;

  optixTrace(params.handle, origin, dir, tmin, tmax, 0,
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_NONE,  // OPTIX_RAY_FLAG_NONE,
        SURFACE_RAY_TYPE,     // SBT offset
        RAY_TYPE_COUNT,       // SBT stride
        SURFACE_RAY_TYPE,     // missSBTIndex
        tmp);

}
