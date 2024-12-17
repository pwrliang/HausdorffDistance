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
    hd::details::LaunchParamsNN<FLOAT_TYPE, 2> params;

extern "C" __global__ void __intersection__nn_2d() {
  auto point_a_id = optixGetPayload_0();
  auto point_b_id = optixGetPrimitiveIndex();
  auto &aabb = params.aabbs[point_b_id];
  const auto& point_a = params.points_a[point_a_id];
  const auto& point_b = params.points_b[point_b_id];

  auto dist2 = hd::EuclideanDistance2(point_a, point_b);
  uint2 cmin2_storage{optixGetPayload_1(), optixGetPayload_2()};
  double cmin2;

  hd::unpack64(cmin2_storage.x, cmin2_storage.y, &cmin2);

  atomicAdd(params.total_hits, 1);

  if (dist2 < cmin2) {
    cmin2 = dist2;
    hd::pack64(&cmin2, cmin2_storage.x, cmin2_storage.y);
    optixSetPayload_1(cmin2_storage.x);
    optixSetPayload_2(cmin2_storage.y);
  }
//    volatile auto *p_cmax2 = params.cmax2;
  auto cmax2 = atomicMax(params.cmax2, static_cast<FLOAT_TYPE>(0));
//  auto cmax2 = *p_cmax2;
  if (cmin2 < cmax2) {
    optixReportIntersection(0, 0);
  }
}


extern "C" __global__ void __anyhit__nn_2d() {
  // printf("terminate ray\n");
  optixTerminateRay();
}

extern "C" __global__ void __raygen__nn_2d() {
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

    double cmin2 = std::numeric_limits<double>::max(); // params.closest_dist[point_id_a]; //std::numeric_limits<double>::max();
    uint2 cmin2_storage;
    hd::pack64(&cmin2, cmin2_storage.x, cmin2_storage.y);

    optixTrace(params.handle, origin, dir, tmin, tmax, 0,
               OptixVisibilityMask(255),
               OPTIX_RAY_FLAG_NONE,  // OPTIX_RAY_FLAG_NONE,
               SURFACE_RAY_TYPE,     // SBT offset
               RAY_TYPE_COUNT,       // SBT stride
               SURFACE_RAY_TYPE,     // missSBTIndex
               point_id_a, cmin2_storage.x, cmin2_storage.y);

    hd::unpack64(cmin2_storage.x, cmin2_storage.y, &cmin2);

    if (cmin2 != std::numeric_limits<double>::max()) {
      atomicMax(params.cmax2, static_cast<FLOAT_TYPE>(cmin2));
    } else {
      params.out_queue.Append(point_id_a);
    }
  }
}
