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
extern "C" __constant__ hd::details::LaunchParamsNNUniformGrid<FLOAT_TYPE, 2>
    params;

extern "C" __global__ void __intersection__nn_uniform_grid_2d() {
  auto point_a_id = optixGetPayload_0();
  auto n_hits = optixGetPayload_1();
  auto mbr_id = optixGetPrimitiveIndex();
  const auto& point_a = params.points_a[point_a_id];
  const auto& mbr_b = params.mbrs_b[mbr_id];
  auto begin = params.prefix_sum[mbr_id];
  auto end = params.prefix_sum[mbr_id + 1];
  auto radius = params.radius;
  auto min_dist2 = mbr_b.GetMinDist2(point_a);
  auto max_dist2 = mbr_b.GetMaxDist2(point_a);
  auto update_cmin2 = [](FLOAT_TYPE dist2) {
    FLOAT_TYPE cmin2;
    if (sizeof(FLOAT_TYPE) == sizeof(float)) {
      auto cmin2_storage = optixGetPayload_2();
      cmin2 = *reinterpret_cast<FLOAT_TYPE*>(&cmin2_storage);

      if (dist2 < cmin2) {
        cmin2 = dist2;
        cmin2_storage = *reinterpret_cast<unsigned int*>(&cmin2);
        optixSetPayload_2(cmin2_storage);
      }
    } else {
      uint2 cmin2_storage{optixGetPayload_2(), optixGetPayload_3()};
      hd::unpack64(cmin2_storage.x, cmin2_storage.y, &cmin2);

      if (dist2 < cmin2) {
        cmin2 = dist2;
        hd::pack64(&cmin2, cmin2_storage.x, cmin2_storage.y);
        optixSetPayload_2(cmin2_storage.x);
        optixSetPayload_3(cmin2_storage.y);
      }
    }
  };
  if (params.hits_counters != nullptr) {
    params.hits_counters[optixGetLaunchIndex().x]++;
  }
  // this box is out of search radius
  // This improves the performance by a lot
  if (min_dist2 > radius * radius) {
    return;
  }
  // max dist is less than cmax, cannot produce a greater dist, so break
  // this almost does not improve performance
  if (max_dist2 <= *params.cmax2) {
    update_cmin2(max_dist2);
    optixReportIntersection(0, 0);
  }

  optixSetPayload_1(n_hits + 1);

  for (auto offset = begin; offset < end; ++offset) {
    auto point_b_id = params.point_b_ids[offset];
    const auto& point_b = params.points_b[point_b_id];
    auto dist2 = hd::EuclideanDistance2(point_a, point_b);

    if (dist2 <= radius * radius) {
      update_cmin2(dist2);
    }

    if (dist2 <= *params.cmax2 || n_hits > params.max_hit) {
      optixReportIntersection(0, 0);
    }
  }
}

extern "C" __global__ void __anyhit__nn_uniform_grid_2d() { optixTerminateRay(); }

extern "C" __global__ void __raygen__nn_uniform_grid_2d() {
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

    auto cmin2 = std::numeric_limits<FLOAT_TYPE>::max();
    unsigned int n_hits = 0;

    if (sizeof(FLOAT_TYPE) == sizeof(float)) {
      auto cmin2_storage = *reinterpret_cast<unsigned int*>(&cmin2);
      optixTrace(params.handle, origin, dir, tmin, tmax, 0,
                 OptixVisibilityMask(255),
                 OPTIX_RAY_FLAG_NONE,  // OPTIX_RAY_FLAG_NONE,
                 SURFACE_RAY_TYPE,     // SBT offset
                 RAY_TYPE_COUNT,       // SBT stride
                 SURFACE_RAY_TYPE,     // missSBTIndex
                 point_id_a, n_hits, cmin2_storage);
      cmin2 = *reinterpret_cast<FLOAT_TYPE*>(&cmin2_storage);
    } else {
      uint2 cmin2_storage;
      hd::pack64(&cmin2, cmin2_storage.x, cmin2_storage.y);
      optixTrace(params.handle, origin, dir, tmin, tmax, 0,
                 OptixVisibilityMask(255),
                 OPTIX_RAY_FLAG_NONE,  // OPTIX_RAY_FLAG_NONE,
                 SURFACE_RAY_TYPE,     // SBT offset
                 RAY_TYPE_COUNT,       // SBT stride
                 SURFACE_RAY_TYPE,     // missSBTIndex
                 point_id_a, n_hits, cmin2_storage.x, cmin2_storage.y);
      hd::unpack64(cmin2_storage.x, cmin2_storage.y, &cmin2);
    }
    if (params.n_hits != nullptr) {
      atomicAdd(params.n_hits, n_hits);
    }

    if (n_hits > params.max_hit) {
      if (params.term_queue.capacity()) {
        params.term_queue.Append(point_id_a);
      }
    } else {
      if (cmin2 != std::numeric_limits<FLOAT_TYPE>::max()) {
        atomicMax(params.cmax2, cmin2);
      } else {
        if (params.term_queue.capacity()) {
          params.miss_queue.Append(point_id_a);
        }
      }
    }
  }
}
