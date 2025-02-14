#include <cuda.h>
#include <cuda_runtime.h>
#include <optix.h>
#include <optix_device.h>

#include "rt/launch_parameters.h"
#include "distance.h"
#include "grid.h"
#include "utils/derived_atomic_functions.h"
#include "utils/helpers.h"

enum { SURFACE_RAY_TYPE = 0, RAY_TYPE_COUNT };
// FLOAT_TYPE is defined by CMakeLists.txt
extern "C" __constant__
    hd::details::LaunchParamsNNGrid<FLOAT_TYPE, 2> params;
#if 0
extern "C" __global__ void __intersection__nn_grid_2d() {
  using point_t = typename decltype(params)::point_t;
  auto point_a_id = optixGetPayload_0();
  auto skip_idx = optixGetPayload_1();
  auto cell_idx = optixGetPrimitiveIndex();
  const auto& point_a = params.points_a[point_a_id];
  // const auto& point_b = params.points_b[point_b_id];
  const auto& grid = params.grid;
  auto radius = params.radius;

  if (params.n_hits != nullptr) {
  	atomicAdd(params.n_hits, 1);
  }


  auto pids_begin = grid.begin(cell_idx);
  auto pids_end = grid.end(cell_idx);

  for (auto pid = pids_begin; pid != pids_end; ++pid) {
    const auto &point_b = params.points_b[pid];
    FLOAT_TYPE cmin2;
    auto dist2 = hd::EuclideanDistance2(point_a, point_b);
    if (params.n_compared_pairs != nullptr) {
      atomicAdd(params.n_compared_pairs, 1);
    }

    // point is within the circle
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

    optixSetPayload_1(skip_idx + 1);

    auto cmax2 = *params.cmax2;

    if (dist2 < cmax2) {
      if (params.skip_count != nullptr) {
        atomicAdd(params.skip_count, 1);
      }
      if (params.skip_total_idx != nullptr) {
        atomicAdd(params.skip_total_idx, skip_idx);
      }
      optixReportIntersection(0, 0);
    }
  }
}
#endif

extern "C" __global__ void __closesthit__nn_grid_2d() {
	auto tri_id = optixGetPrimitiveIndex();
    atomicAdd(params.n_hits, 1);
	optixSetPayload_0(tri_id);
}

extern "C" __global__ void __raygen__nn_grid_2d() {
  using coord_min_max_t = typename cuda_vec<FLOAT_TYPE, 2>::type;
  const auto& in_queue = params.in_queue;
  const auto& grid = params.grid;
  const auto& points_a = params.points_a;
  auto grid_size = grid.get_grid_size();
  auto grid_extent_x = grid.get_cell_extent(0);
  float tmin = 0;
  float tmax = FLT_MAX;

  for (auto i = optixGetLaunchIndex().x; i < in_queue.size();
       i += optixGetLaunchDimensions().x) {
    auto point_id_a = in_queue[i];
    const auto& point_a = points_a[i];
    auto cell_idx = grid.CalculateCellIdx(point_a);
    // the cell pos where the query point residents in
    auto cell_pos = grid.DecodeCellIdx(cell_idx);

    FLOAT_TYPE cmin2;
    FLOAT_TYPE cmax2 = 0; // cmax2 > 0 if hit something

	auto cast_ray = [&]__device__(const uint2& ray_cell_pos, const float3& dir) {
		auto center = grid.GetCellBounds(ray_cell_pos).get_center();

		float3 origin;
		origin.x = center.x;
		origin.y = center.y;
		origin.z = 0;

		auto tri_id = std::numeric_limits<unsigned int>::max();

		optixTrace(params.handle, origin, dir, tmin, tmax, 0,
			OptixVisibilityMask(255),
			OPTIX_RAY_FLAG_NONE,  // OPTIX_RAY_FLAG_NONE,
			SURFACE_RAY_TYPE,     // SBT offset
			RAY_TYPE_COUNT,       // SBT stride
			SURFACE_RAY_TYPE,     // missSBTIndex
			tri_id);

		if (tri_id != std::numeric_limits<unsigned int>::max()) {
        	auto ch_cell_idx = params.tri_to_cell_idx[tri_id];
            const auto& mbr = grid.get_mbr(ch_cell_idx);

            auto min2 = mbr.GetMinDist2(point_a);
			auto max2 = mbr.GetMaxDist2(point_a);

            if (cmax2 == 0) { // first hit
              cmin2 = min2;
              cmax2 = max2;
            } else if (min2 < cmax2) { // has distance overlap
              cmin2 = std::min(cmin2, min2);
              cmax2 = std::max(cmax2, max2);
            }
        }
	};

    for(int cell_x_offset = 0; cell_x_offset < grid_size; cell_x_offset++) {
      auto cell_x_low = (int) cell_pos.x - cell_x_offset;
      auto cell_x_high = cell_pos.x + cell_x_offset;
      auto extent_x = cell_x_offset * grid_extent_x;

      if (cell_x_low >= 0) {
        // up
        cast_ray(uint2{(unsigned int) cell_x_low, cell_pos.y}, float3{0, 1, 0});
        // down
        cast_ray(uint2{(unsigned int) cell_x_low, cell_pos.y}, float3{0, -1, 0});
      }

      if (cell_x_high < grid_size) {
        cast_ray(uint2{cell_x_high, cell_pos.y}, float3{0, 1, 0});
        cast_ray(uint2{cell_x_high, cell_pos.y}, float3{0, -1, 0});
      }
      // cannot produce a lower bound
      if (extent_x >= params.radius || cmax2 > 0 && extent_x > cmax2) {
        break;
      }
    }
  }
}
