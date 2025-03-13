#include <cuda.h>
#include <cuda_runtime.h>
#include <optix.h>
#include <optix_device.h>

#include "distance.h"
#include "rt/launch_parameters.h"
#include "utils/derived_atomic_functions.h"
#include "utils/helpers.h"

enum { SURFACE_RAY_TYPE = 0, RAY_TYPE_COUNT };
// FLOAT_TYPE is defined by CMakeLists.txt
extern "C" __constant__ hd::details::LaunchParamsNNTensor<FLOAT_TYPE, 2> params;
#if 0
extern "C" __global__ void __intersection__nn_tensor_2d() {
  using coopvec_t = OptixCoopVec<float, TENSOR_2D_BATCH_SIZE * 2>;
  using point_t = typename decltype(params)::point_t;
  auto ray_idx = optixGetPayload_0();
  auto point_a_id = params.in_queue[ray_idx];
  auto n_hits = optixGetPayload_1();
  auto point_b_id = optixGetPrimitiveIndex();

  const auto& point_a = params.points_a[point_a_id];
  const auto& point_b = params.points_b[point_b_id];
  auto radius = params.radius;

  FLOAT_TYPE min_x = point_b.x - radius;
  FLOAT_TYPE max_x = point_b.x + radius;
  FLOAT_TYPE min_y = point_b.y - radius;
  FLOAT_TYPE max_y = point_b.y + radius;

  if (point_a.x >= min_x && point_a.x <= max_x && point_a.y >= min_y &&
      point_a.y <= max_y) {
    optixSetPayload_1(n_hits + 1);
    FLOAT_TYPE cmin2;

    auto offset = n_hits % TENSOR_2D_BATCH_SIZE;
    // points_a_batched has been prefilled
    params.points_b_batched[ray_idx * TENSOR_2D_BATCH_SIZE + offset] = point_b;

    if (n_hits > 0 && offset == 0) {
      auto points_a_batched = optixCoopVecLoad<coopvec_t>(
          &params.points_a_batched[ray_idx * TENSOR_2D_BATCH_SIZE]);
      auto points_b_batched = optixCoopVecLoad<coopvec_t>(
          &params.points_b_batched[ray_idx * TENSOR_2D_BATCH_SIZE]);
      // x_a1 - x_b1, y_a1 - y_b1, x_a2 - x_b2, y_a2 - y_b2...
      auto points_a_b = optixCoopVecSub(points_a_batched, points_b_batched);
      // (x_a1 - x_b1)^2, (y_a1 - y_b1)^2, (x_a2 - x_b2)^2, (y_a2 - y_b2)^2...
      points_a_b = optixCoopVecMul(points_a_b, points_a_b);

      auto cmin2_storage = optixGetPayload_2();
      cmin2 = *reinterpret_cast<FLOAT_TYPE*>(&cmin2_storage);
      auto cmax2 = *params.cmax2;

      for (auto i = 0; i < TENSOR_2D_BATCH_SIZE; i++) {
        auto dist2 = points_a_b[i * 2] + points_a_b[i * 2 + 1];
        // dist2 and radius*radius to vectors, use step function to filter out
        // out-of-circle distances
        if (dist2 <= radius * radius) {
          if (dist2 <= cmax2) {
            optixReportIntersection(0, 0);
          }
          if (dist2 < cmin2) {
            cmin2 = dist2;
          }
        }
      }
      cmin2_storage = *reinterpret_cast<unsigned int*>(&cmin2);
      optixSetPayload_2(cmin2_storage);
    }

    if (n_hits > params.max_hit) {
      optixReportIntersection(0, 0);
    }
  }
}

extern "C" __global__ void __anyhit__nn_tensor_2d() { optixTerminateRay(); }

extern "C" __global__ void __raygen__nn_tensor_2d() {
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

    auto cmin2_storage = *reinterpret_cast<unsigned int*>(&cmin2);
    optixTrace(params.handle, origin, dir, tmin, tmax, 0,
               OptixVisibilityMask(255),
               OPTIX_RAY_FLAG_NONE,  // OPTIX_RAY_FLAG_NONE,
               SURFACE_RAY_TYPE,     // SBT offset
               RAY_TYPE_COUNT,       // SBT stride
               SURFACE_RAY_TYPE,     // missSBTIndex
               i, n_hits, cmin2_storage);
    cmin2 = *reinterpret_cast<FLOAT_TYPE*>(&cmin2_storage);
    atomicAdd(params.n_hits, n_hits);

    if (n_hits > params.max_hit) {
      params.term_queue.Append(point_id_a);
    } else {
      auto radius = params.radius;
      // handle remaining accumulated points
      // TODO: Do Tensor cores help, here?
      for (int offset = 0; offset < n_hits % TENSOR_2D_BATCH_SIZE; offset++) {
        auto dist2 = EuclideanDistance2(point_a, params.points_a_b[i * 2]);
        // dist2 and radius*radius to vectors, use step function to filter out
        // out-of-circle distances
        if (dist2 <= radius * radius) {
          if (dist2 <= *params.cmax2) {
            break;
          }
          if (dist2 < cmin2) {
            cmin2 = dist2;
          }
        }
      }

      if (cmin2 != std::numeric_limits<FLOAT_TYPE>::max()) {
        atomicMax(params.cmax2, cmin2);
      } else {
        params.miss_queue.Append(point_id_a);
      }
    }
  }
}
#endif