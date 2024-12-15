#ifndef HAUSDORFF_DISTANCE_H
#define HAUSDORFF_DISTANCE_H
#include <glog/logging.h>
#include <thrust/device_vector.h>

#include "distance.h"
#include "rt/reusable_buffer.h"
#include "rt/rt_engine.h"
#include "sampler.h"
#include "utils/bitset.h"
#include "utils/derived_atomic_functions.h"
#include "utils/helpers.h"
#include "utils/queue.h"
#include "utils/stream.h"
#include "utils/type_traits.h"
#include "utils/util.h"
#include "utils/stopwatch.h"

namespace hd {
namespace details {

DEV_HOST_INLINE OptixAabb GetOptixAABB(float2 p, float radius) {
  OptixAabb aabb;
  aabb.minX = p.x - radius;
  aabb.maxX = p.x + radius;
  aabb.minY = p.y - radius;
  aabb.maxY = p.y + radius;
  aabb.minZ = aabb.maxZ = 0;
  return aabb;
}

DEV_HOST_INLINE OptixAabb GetOptixAABB(float3 p, float radius) {
  OptixAabb aabb;
  aabb.minX = p.x - radius;
  aabb.maxX = p.x + radius;
  aabb.minY = p.y - radius;
  aabb.maxY = p.y + radius;
  aabb.minZ = p.z - radius;
  aabb.maxZ = p.z + radius;
  return aabb;
}

DEV_HOST_INLINE OptixAabb GetOptixAABB(double2 p, double radius) {
  OptixAabb aabb;
  aabb.minX = next_float_from_double(p.x - radius, -1, 2);
  aabb.maxX = next_float_from_double(p.x + radius, 1, 2);
  aabb.minY = next_float_from_double(p.y - radius, -1, 2);
  aabb.maxY = next_float_from_double(p.y + radius, 1, 2);
  aabb.minZ = aabb.maxZ = 0;
  return aabb;
}

DEV_HOST_INLINE OptixAabb GetOptixAABB(double3 p, double radius) {
  OptixAabb aabb;
  aabb.minX = next_float_from_double(p.x - radius, -1, 2);
  aabb.maxX = next_float_from_double(p.x + radius, 1, 2);
  aabb.minY = next_float_from_double(p.y - radius, -1, 2);
  aabb.maxY = next_float_from_double(p.y + radius, 1, 2);
  aabb.minZ = next_float_from_double(p.z - radius, -1, 2);
  aabb.maxZ = next_float_from_double(p.z + radius, 1, 2);
  return aabb;
}

std::vector<float> GetScaleMatrix(float scale) {
  std::vector<float> mat = {1.0f * scale, 0.0f,         0.0f,         0.0f,
                            0.0f,         1.0f * scale, 0.0f,         0.0f,
                            0.0f,         0.0f,         1.0f * scale, 0.0f};
  return mat;
}

template <typename POINT_T>
__global__ void CalculateNNDist2(ArrayView<POINT_T> points_a,
                                 ArrayView<POINT_T> points_b,
                                 float* min_dist2) {
  auto warp_id = TID_1D / WARP_SIZE;
  auto n_warps = TOTAL_THREADS_1D / WARP_SIZE;
  auto lane_id = threadIdx.x % WARP_SIZE;

  for (uint32_t point_id_a = warp_id; point_id_a < points_a.size();
       point_id_a += n_warps) {
    for (uint32_t point_id_b = lane_id; point_id_b < points_b.size();
         point_id_b += WARP_SIZE) {
      float dist2 =
          EuclideanDistance2(points_a[point_id_a], points_b[point_id_b]);
      if (dist2 != 0) {
        atomicMin(min_dist2, dist2);
      }
    }
  }
}

}  // namespace details

struct HausdorffDistanceConfig {
  const char* ptx_root;
  bool fast_build = false;
  bool compact = false;
  float sample_rate = 0.01;
};

template <typename COORD_T, int N_DIMS>
class HausdorffDistance {
  using coord_t = COORD_T;
  using point_t = typename cuda_vec<COORD_T, N_DIMS>::type;

 public:
  HausdorffDistance() = default;

  void Init(const HausdorffDistanceConfig& hd_config) {
    config_ = hd_config;
    auto rt_config = details::get_default_rt_config(hd_config.ptx_root);
    rt_engine_.Init(rt_config);
  }

  template <typename IT>
  void SetPointsTo(const Stream& stream, IT begin, IT end) {
    points_b_.assign(begin, end);

    auto n_points = points_b_.size();
    auto mem_bytes = rt_engine_.EstimateMemoryUsageForAABB(
        n_points, config_.fast_build, config_.compact);

    buffer_.Init(mem_bytes * 1.5);
    buffer_.Clear();
  }

  template <typename IT>
  COORD_T CalculateDistanceFrom(const Stream& stream, IT begin, IT end) {
    points_a_.assign(begin, end);
    Stopwatch sw;
    sw.start();
    auto radius = CalculateInitialRadius(stream);
    sw.stop();

    LOG(INFO) << "init_radius: " << radius << " Time: " << sw.ms() << " ms";
    std::vector<OptixTraversableHandle> handles{gas_handle_};

    in_queue_.Init(points_a_.size());
    out_queue_.Init(points_a_.size());

    in_queue_.Clear(stream.cuda_stream());
    out_queue_.Clear(stream.cuda_stream());

    auto d_in_queue = in_queue_.DeviceObject();

    thrust::for_each(thrust::cuda::par.on(stream.cuda_stream()),
                     thrust::make_counting_iterator<uint32_t>(0),
                     thrust::make_counting_iterator<uint32_t>(points_a_.size()),
                     [=] __device__(uint32_t point_id) mutable {
                       d_in_queue.Append(point_id);
                     });

    uint32_t in_size = points_a_.size();
    cmax2_.set(stream.cuda_stream(), 0);

    sw.start();
    auto gas_handle = BuildBVH(stream, points_b_, radius);
    stream.Sync();
    sw.stop();

    LOG(INFO) << "Init BVH Time: " << sw.ms() << " ms";

    sw.start();
    while (in_size > 0) {
      details::LaunchParamsNN<COORD_T, N_DIMS> launch_params;

      launch_params.in_queue = in_queue_.DeviceObject();
      launch_params.out_queue = out_queue_.DeviceObject();
      launch_params.points_a = ArrayView<point_t>(points_a_);
      launch_params.points_b = ArrayView<point_t>(points_b_);
      launch_params.handle = gas_handle;
      launch_params.aabbs = ArrayView<OptixAabb>(aabbs_);
      launch_params.cmax2 = cmax2_.data();

      details::ModuleIdentifier mod = details::NUM_MODULE_IDENTIFIERS;

      if (typeid(COORD_T) == typeid(float)) {
        if (N_DIMS == 2) {
          mod = details::MODULE_ID_FLOAT_NN_2D;
        } else if (N_DIMS == 3) {
          mod = details::MODULE_ID_FLOAT_NN_3D;
        }
      } else if (typeid(COORD_T) == typeid(double)) {
        if (N_DIMS == 2) {
          mod = details::MODULE_ID_DOUBLE_NN_2D;
        } else if (N_DIMS == 3) {
          mod = details::MODULE_ID_DOUBLE_NN_3D;
        }
      }

      dim3 dims{1, 1, 1};

      dims.x = in_size;
      rt_engine_.CopyLaunchParams(stream.cuda_stream(), launch_params);
      rt_engine_.Render(stream.cuda_stream(), mod, dims);

      // buffer_.SetTail(tail);
      // LOG(INFO) << "radius: " << radius << " in_size: " << in_size
                // << " out_size: " << out_queue_.size(stream.cuda_stream());
      in_queue_.Clear(stream.cuda_stream());
      in_queue_.Swap(out_queue_);
      radius *= 2;

      in_size = in_queue_.size(stream.cuda_stream());
      if (in_size > 0) {
        gas_handle = UpdateBVH(stream, gas_handle, points_b_, radius);
      }
    }
    sw.stop();

    LOG(INFO) << "Calculate Time " << sw.ms() << " ms";
    return sqrt(cmax2_.get(stream.cuda_stream()));
  }

  OptixTraversableHandle BuildBVH(const Stream& stream,
                                  ArrayView<point_t> points, COORD_T radius) {
    aabbs_.resize(points.size());
    thrust::transform(thrust::cuda::par.on(stream.cuda_stream()),
                      points.begin(), points.end(), aabbs_.begin(),
                      [=] __device__(const point_t& p) {
                        return details::GetOptixAABB(p, radius);
                      });
    return rt_engine_.BuildAccelCustom(stream.cuda_stream(),
                                       ArrayView<OptixAabb>(aabbs_), buffer_,
                                       config_.fast_build, config_.compact);
  }

  OptixTraversableHandle UpdateBVH(const Stream& stream,
                                   OptixTraversableHandle handle,
                                   ArrayView<point_t> points, COORD_T radius) {
    aabbs_.resize(points.size());
    thrust::transform(thrust::cuda::par.on(stream.cuda_stream()),
                      points.begin(), points.end(), aabbs_.begin(),
                      [=] __device__(const point_t& p) {
                        return details::GetOptixAABB(p, radius);
                      });
    return rt_engine_.UpdateAccelCustom(stream.cuda_stream(), handle,
                                        ArrayView<OptixAabb>(aabbs_), buffer_,
                                        0, config_.fast_build, config_.compact);
  }

  COORD_T CalculateInitialRadius(const Stream& stream) {
    size_t sample_size_a = points_a_.size() * config_.sample_rate + 1;
    size_t sample_size_b = points_b_.size() * config_.sample_rate + 1;
    thrust::device_vector<point_t> samples_a, samples_b;
    auto sample_size = std::max(sample_size_a, sample_size_b);

    sampler_.Init(sample_size);

    sampler_.Sample(stream.cuda_stream(), ArrayView<point_t>(points_a_),
                    sample_size_a, samples_a);
    sampler_.Sample(stream.cuda_stream(), ArrayView<point_t>(points_b_),
                    sample_size_b, samples_b);

    ArrayView<point_t> v_samples_a(samples_a);
    ArrayView<point_t> v_samples_b(samples_b);

    if (v_samples_a.size() < v_samples_b.size()) {
      v_samples_a.Swap(v_samples_b);
    }

    int grid_size, block_size;
    SharedValue<float> dist2;

    dist2.set(stream.cuda_stream(), FLT_MAX);

    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(
        &grid_size, &block_size, details::CalculateNNDist2<point_t>, 0,
        reinterpret_cast<int>(MAX_BLOCK_SIZE)));

    details::
        CalculateNNDist2<<<grid_size, block_size, 0, stream.cuda_stream()>>>(
            v_samples_a, v_samples_b, dist2.data());
    return sqrt(dist2.get(stream.cuda_stream()));
  }

 private:
  HausdorffDistanceConfig config_;
  thrust::device_vector<point_t> points_a_;
  thrust::device_vector<point_t> points_b_;
  thrust::device_vector<OptixAabb> aabbs_;
  OptixTraversableHandle gas_handle_;
  Queue<uint32_t> in_queue_, out_queue_;
  Sampler sampler_;
  ReusableBuffer buffer_;
  details::RTEngine rt_engine_;
  SharedValue<COORD_T> cmax2_;
};
}  // namespace hd

#endif  // HAUSDORFF_DISTANCE_H
