#include <optix.h>
// #include <optix_function_table_definition.h>  // for g_optixFunctionTable
#include <thrust/device_vector.h>

#include <vector>

#include "play.cuh"
#include "rt/launch_parameters.h"
#include "rt/rt_engine.h"
#include "run_config.h"

namespace hd {
std::vector<OptixAabb> GetAabbs1() {
  std::vector<OptixAabb> aabbs;
  std::vector<float> begins = {0, 0.1,  0.3, 0.4, 0.2,};

  for (auto begin : begins) {
    OptixAabb aabb;
    aabb.minX = begin;
    aabb.maxX = 1 - begin;
    aabb.minY = begin;
    aabb.maxY = 1 - begin;
    aabb.minZ = begin;
    aabb.maxZ = 1 - begin;
    aabbs.push_back(aabb);
  }

  return aabbs;
}

void Play(const RunConfig& config) {
  std::vector<OptixAabb> aabbs = GetAabbs1();
  thrust::device_vector<OptixAabb> d_aabbs = aabbs;
  details::RTEngine rt_engine;
  ReusableBuffer buffer;
  bool fast_build = false;
  bool compact = false;
  std::string ptx_root = config.exec_path + "/ptx";
  auto rt_config = details::get_default_rt_config(ptx_root);

  rt_engine.Init(rt_config);

  auto mem_bytes =
      rt_engine.EstimateMemoryUsageForAABB(d_aabbs.size(), fast_build, compact);

  buffer.Init(mem_bytes * 1.5);
  buffer.Clear();

  Stream stream;

  auto handle = rt_engine.BuildAccelCustom(stream.cuda_stream(),
                                           ArrayView<OptixAabb>(d_aabbs),
                                           buffer, fast_build, compact);

  details::LaunchParamsPlay params;
  dim3 dims;
  details::ModuleIdentifier mod = details::MODULE_ID_PLAY;

  params.handle = handle;

  rt_engine.CopyLaunchParams(stream.cuda_stream(), params);
  dims = dim3{1, 1, 1};
  rt_engine.Render(stream.cuda_stream(), mod, dims);
  stream.Sync();
}
}  // namespace hd