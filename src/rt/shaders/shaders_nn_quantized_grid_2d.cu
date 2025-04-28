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
extern "C" __constant__ hd::details::LaunchParamsNNQuantizedGrid<FLOAT_TYPE, 2>
    params;

extern "C" __global__ void __intersection__nn_quantized_grid_2d() {
  auto point_a_id = optixGetPayload_0();
  auto cell_idx = optixGetPrimitiveIndex();
  const auto& point_a = params.points_a[point_a_id];
  auto begin = params.prefix_sum[cell_idx];
  auto end = params.prefix_sum[cell_idx + 1];
  auto radius = params.radius;
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

  auto quantized_point_a = params.grid.QuantizePoint(point_a);
  auto quantized_point_b = params.representative_points[cell_idx];
  auto quantized_dist2 =
      hd::EuclideanDistance2(quantized_point_a, quantized_point_b);

  // printf("quantized dist2 %d, radius2 %f pa (%d,%d,%d) pb (%d,%d,%d)\n",
  //        quantized_dist2, radius * radius, quantized_point_a.x,
  //        quantized_point_a.y, quantized_point_a.z, quantized_point_b.x,
  //        quantized_point_b.y, quantized_point_b.z);

  if (quantized_dist2 <= radius * radius) {
    optixSetPayload_1(1);  // hit = 1

    for (auto offset = begin; offset < end; ++offset) {
      auto point_b_id = params.point_b_ids[offset];
      const auto& point_b = params.points_b[point_b_id];
      auto dist2 = hd::EuclideanDistance2(point_a, point_b);

      update_cmin2(dist2);
    }
  }
}

extern "C" __global__ void __raygen__nn_quantized_grid_2d() {
  const auto& in_queue = params.in_queue;
  const auto& grid = params.grid;
  float tmin = 0;
  float tmax = FLT_MIN;
  using query_t = decltype(params)::Query;

  for (auto i = optixGetLaunchIndex().x; i < in_queue.size();
       i += optixGetLaunchDimensions().x) {
    const auto& query = in_queue[i];
    auto point_id_a = query.point_id;
    auto iter = query.iter;
    const auto& point_a = params.points_a[point_id_a];
    auto quantized_point_a = grid.QuantizePoint(point_a);

    float3 origin;
    origin.x = quantized_point_a.x;
    origin.y = quantized_point_a.y;
    origin.z = 0;
    float3 dir = {0, 0, 1};

    auto cmin2 = std::numeric_limits<FLOAT_TYPE>::max();
    unsigned int hit = 0;

    if (sizeof(FLOAT_TYPE) == sizeof(float)) {
      auto cmin2_storage = *reinterpret_cast<unsigned int*>(&cmin2);
      optixTrace(params.handle, origin, dir, tmin, tmax, 0,
                 OptixVisibilityMask(255),
                 OPTIX_RAY_FLAG_NONE,  // OPTIX_RAY_FLAG_NONE,
                 SURFACE_RAY_TYPE,     // SBT offset
                 RAY_TYPE_COUNT,       // SBT stride
                 SURFACE_RAY_TYPE,     // missSBTIndex
                 point_id_a, hit, cmin2_storage);
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
                 point_id_a, hit, cmin2_storage.x, cmin2_storage.y);
      hd::unpack64(cmin2_storage.x, cmin2_storage.y, &cmin2);
    }

    if (hit) {
      // According to the paper, the distance has an error bound, where the
      // lower bound is nr - 2r, and the upper bound nr + r,
      // so we keep 3 iterations of points
      if (iter < 3) {
        params.hit_queue.Append(query_t(point_id_a, iter + 1));
      }
      if (cmin2 != std::numeric_limits<FLOAT_TYPE>::max()) {
        atomicMax(params.cmax2, cmin2);
      }
    } else {
      params.miss_queue.Append(point_id_a);
    }
  }
}
