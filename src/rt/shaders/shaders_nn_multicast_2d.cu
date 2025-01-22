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
    hd::details::LaunchParamsNNMultiCast<FLOAT_TYPE, 2> params;

template<typename COORD_T, int N_DIMS>
__device__ __forceinline__ typename cuda_vec<COORD_T, N_DIMS>::type
ScalePoint(const typename cuda_vec<COORD_T, N_DIMS>::type& p,
           const hd::Mbr<COORD_T, N_DIMS>& mbr,
           int partition) {
  typename cuda_vec<COORD_T, N_DIMS>::type scaled_p;

  for (int dim = 0; dim < N_DIMS; ++dim) {
    auto val = reinterpret_cast<const COORD_T*>(&p.x)[dim];
    auto lower = mbr.lower(dim);
    auto upper = mbr.upper(dim);

    assert(val >= lower && val <= upper);
    auto norm_val = (val - lower) / (upper - lower);

    assert(norm_val >= 0 && norm_val <= 1);

    reinterpret_cast<COORD_T*>(&scaled_p.x)[dim] = norm_val + partition;
  }
  return scaled_p;
}


extern "C" __global__ void __intersection__nn_multicast_2d() {
  using point_t = typename decltype(params)::point_t;
  auto point_a_id = optixGetPayload_0();
  auto skip_idx = optixGetPayload_1();
  auto point_b_id = optixGetPrimitiveIndex();
  const auto& point_a = params.points_a[point_a_id];
  const auto& point_b = params.points_b[point_b_id];
  auto radius = params.radius * sqrt(2); // Actuall AABB size
  auto partition_hit = point_b_id % optixGetLaunchDimensions().y;

  atomicAdd(params.n_hits, 1);

  if (optixGetLaunchIndex().y == partition_hit) {
    if (point_a.x >= point_b.x - radius && point_a.x <= point_b.x + radius &&
        point_a.y >= point_b.y - radius && point_a.y <= point_b.y + radius) {
      FLOAT_TYPE cmin2;
      auto dist2 = hd::EuclideanDistance2(point_a, point_b);
      atomicAdd(params.n_compared_pairs, 1);

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

      auto cmax2 = *params.cmax2;
      optixSetPayload_1(skip_idx + 1);

      if (dist2 <= cmax2) {
        atomicAdd(params.skip_count, 1);
        atomicAdd(params.skip_total_idx, skip_idx);
        optixReportIntersection(0, 0);
      }
    }
  }
}


extern "C" __global__ void __anyhit__nn_multicast_2d() {
  optixTerminateRay();
}

extern "C" __global__ void __raygen__nn_multicast_2d() {
  const auto& in_queue = params.in_queue;
  float tmin = 0;
  float tmax = FLT_MIN;

  for (auto i = optixGetLaunchIndex().x; i < in_queue.size();
       i += optixGetLaunchDimensions().x) {
    unsigned partition = optixGetLaunchIndex().y;
    unsigned int point_id_a = in_queue[i];
    const auto& point_a = params.points_a[point_id_a];
    auto scaled_point_a = ScalePoint(point_a, params.mbr, partition);

    float3 origin;
    origin.x = scaled_point_a.x;
    origin.y = scaled_point_a.y;
    origin.z = 0;
    float3 dir = {0, 0, 1};

    auto cmin2 = std::numeric_limits<FLOAT_TYPE>::max();
    unsigned int skip_idx = 0;

    if (sizeof(FLOAT_TYPE) == sizeof(float)) {
      auto cmin2_storage = *reinterpret_cast<unsigned int*>(&cmin2);
      optixTrace(params.handle, origin, dir, tmin, tmax, 0,
            OptixVisibilityMask(255),
            OPTIX_RAY_FLAG_NONE,  // OPTIX_RAY_FLAG_NONE,
            SURFACE_RAY_TYPE,     // SBT offset
            RAY_TYPE_COUNT,       // SBT stride
            SURFACE_RAY_TYPE,     // missSBTIndex
            point_id_a, skip_idx, cmin2_storage);
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
            point_id_a, skip_idx, cmin2_storage.x, cmin2_storage.y);
      hd::unpack64(cmin2_storage.x, cmin2_storage.y, &cmin2);
    }

    // aggreated cmin2
    if (cmin2 != std::numeric_limits<FLOAT_TYPE>::max()) {
      atomicMin(&params.cmin2[i], cmin2);
    }

    // last thread
    if (atomicAdd(&params.thread_counters[i], 1) ==
        optixGetLaunchDimensions().y - 1) {
      auto agg_cmin2 = atomicMax(&params.cmin2[i], 0); // force loading the latest

      // this thread hits nothing
      if (agg_cmin2 == std::numeric_limits<FLOAT_TYPE>::max()) {
        params.out_queue.Append(point_id_a);
      } else {
        atomicMax(params.cmax2, agg_cmin2);
      }
    }
  }
}
