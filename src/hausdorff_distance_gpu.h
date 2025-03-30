#ifndef HAUSDORFF_DISTANCE_GPU_H
#define HAUSDORFF_DISTANCE_GPU_H
#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <thrust/shuffle.h>
#include <utils/type_traits.h>

#include <cub/cub.cuh>
#include <nlohmann/json.hpp>

#include "distance.h"
#include "flags.h"
#include "glog/logging.h"
#include "morton_code.h"
#include "utils/array_view.h"
#include "utils/derived_atomic_functions.h"
#include "utils/launcher.h"
#include "utils/shared_value.h"
#include "utils/stopwatch.h"
#include "utils/stream.h"

namespace hd {

template <typename POINT_T>
typename vec_info<POINT_T>::type CalculateHausdorffDistanceGPU(
    const Stream& stream, thrust::device_vector<POINT_T>& points_a,
    thrust::device_vector<POINT_T>& points_b, int seed, nlohmann::json& stats) {
  using coord_t = typename vec_info<POINT_T>::type;
  Stopwatch sw;
  sw.start();
  thrust::default_random_engine g(seed);
  SharedValue<coord_t> cmax;
  SharedValue<uint32_t> compared_pairs;
  auto* p_cmax = cmax.data();
  auto* p_compared_pairs = compared_pairs.data();
  ArrayView<POINT_T> v_points_a(points_a);
  ArrayView<POINT_T> v_points_b(points_b);

  cudaEvent_t ev_start, ev_shuffle, ev_compute;
  cudaEventCreate(&ev_start);
  cudaEventCreate(&ev_shuffle);
  cudaEventCreate(&ev_compute);

  cmax.set(stream.cuda_stream(), 0);
  compared_pairs.set(stream.cuda_stream(), 0);

  cudaEventRecord(ev_start, stream.cuda_stream());
  thrust::shuffle(thrust::cuda::par.on(stream.cuda_stream()), points_a.begin(),
                  points_a.end(), g);

  thrust::shuffle(thrust::cuda::par.on(stream.cuda_stream()), points_b.begin(),
                  points_b.end(), g);
  cudaEventRecord(ev_shuffle, stream.cuda_stream());

  LaunchKernel(stream, [=] __device__() {
    using BlockReduce = cub::BlockReduce<coord_t, MAX_BLOCK_SIZE>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ bool early_break;
    __shared__ const POINT_T* point_a;

    for (auto i = blockIdx.x; i < v_points_a.size(); i += gridDim.x) {
      auto size_b_roundup =
          div_round_up(v_points_b.size(), blockDim.x) * blockDim.x;
      coord_t cmin = std::numeric_limits<coord_t>::max();
      uint32_t n_pairs = 0;

      if (threadIdx.x == 0) {
        early_break = false;
        point_a = &v_points_a[i];
      }
      __syncthreads();

      for (auto j = threadIdx.x; j < size_b_roundup && !early_break;
           j += blockDim.x) {
        auto d = std::numeric_limits<coord_t>::max();
        if (j < v_points_b.size()) {
          const auto& point_b = v_points_b[j];
          d = EuclideanDistance2(*point_a, point_b);
          n_pairs++;
        }

        auto agg_min = BlockReduce(temp_storage).Reduce(d, cub::Min());

        if (threadIdx.x == 0) {
          cmin = std::min(cmin, agg_min);
          if (cmin <= *p_cmax) {
            early_break = true;
          }
        }
        __syncthreads();
      }
      atomicAdd(p_compared_pairs, n_pairs);
      __syncthreads();
      if (threadIdx.x == 0 && cmin != std::numeric_limits<coord_t>::max()) {
        atomicMax(p_cmax, cmin);
      }
    }
  });
  cudaEventRecord(ev_compute, stream.cuda_stream());
  auto n_compared_pairs = compared_pairs.get(stream.cuda_stream());
  sw.stop();

  float ms;
  cudaEventElapsedTime(&ms, ev_start, ev_shuffle);

  stats["ShuffleTime"] = ms;
  cudaEventElapsedTime(&ms, ev_shuffle, ev_compute);

  stats["ComputeTime"] = ms;
  stats["ComparedPairs"] = n_compared_pairs;

  return sqrt(cmax.get(stream.cuda_stream()));
}

template <typename POINT_T>
typename vec_info<POINT_T>::type CalculateHausdorffDistanceZorderGPU(
    const Stream& stream, thrust::device_vector<POINT_T>& points_a,
    thrust::device_vector<POINT_T>& points_b) {
  using coord_t = typename vec_info<POINT_T>::type;
  constexpr int n_dims = vec_info<POINT_T>::n_dims;
  using mbr_t = Mbr<coord_t, n_dims>;
  SharedValue<mbr_t> mbr;
  SharedValue<coord_t> cmax;
  SharedValue<uint32_t> compared_pairs;
  auto* p_mbr = mbr.data();
  auto* p_cmax = cmax.data();
  auto* p_compared_pairs = compared_pairs.data();
  ArrayView<POINT_T> v_points_a(points_a);
  ArrayView<POINT_T> v_points_b(points_b);

  mbr.set(stream.cuda_stream(), mbr_t());
  cmax.set(stream.cuda_stream(), 0);
  compared_pairs.set(stream.cuda_stream(), 0);

  thrust::for_each(thrust::cuda::par.on(stream.cuda_stream()), points_a.begin(),
                   points_a.end(), [=] __device__(const POINT_T& p) {
                     p_mbr->ExpandAtomic(p);
                   });
  thrust::for_each(thrust::cuda::par.on(stream.cuda_stream()), points_b.begin(),
                   points_b.end(), [=] __device__(const POINT_T& p) {
                     p_mbr->ExpandAtomic(p);
                   });
  auto comp = [=] __device__(const POINT_T& a, const POINT_T& b) {
    auto np_a = p_mbr->Normalize(a);
    auto np_b = p_mbr->Normalize(b);

    return detail::morton_code(np_a) < detail::morton_code(np_b);
  };
  thrust::sort(thrust::cuda::par.on(stream.cuda_stream()), points_a.begin(),
               points_a.end(), comp);
  thrust::sort(thrust::cuda::par.on(stream.cuda_stream()), points_b.begin(),
               points_b.end(), comp);
  stream.Sync();

  SharedValue<uint32_t> dc;
  auto* p_dc = dc.data();

  dc.set(stream.cuda_stream(), points_b.size() / 2);

  LaunchKernel(stream, [=] __device__() {
    using BlockReduce = cub::BlockReduce<coord_t, MAX_BLOCK_SIZE>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ bool early_break;
    __shared__ POINT_T point_a;
    __shared__ int64_t curr_dc;
    __shared__ coord_t cmin;

    for (auto i = blockIdx.x; i < v_points_a.size(); i += gridDim.x) {
      auto size_b_roundup =
          div_round_up(v_points_b.size(), blockDim.x) * blockDim.x;

      if (threadIdx.x == 0) {
        early_break = false;
        point_a = v_points_a[i];
        curr_dc = *p_dc;
        cmin = std::numeric_limits<coord_t>::max();
      }
      __syncthreads();
      int64_t left_begin = std::max(curr_dc - blockDim.x, 0l);
      int64_t left_end = curr_dc;
      int64_t right_begin = curr_dc;
      int64_t right_end =
          std::min(curr_dc + blockDim.x, (int64_t) v_points_b.size());

      while (!early_break &&
             (left_begin < left_end || right_begin < right_end)) {
        // compute with a thread block for the left of the diffusion center
        if (left_begin < left_end) {
          auto d = std::numeric_limits<coord_t>::max();
          unsigned int j;
          for (j = left_begin + threadIdx.x; j < left_end; j += blockDim.x) {
            d = EuclideanDistance2(point_a, v_points_b[j]);
          }
          auto agg_min = BlockReduce(temp_storage).Reduce(d, cub::Min());
          if (threadIdx.x == 0) {
            cmin = std::min(cmin, agg_min);
            atomicAdd(p_compared_pairs, left_end - left_begin);
          }
          __syncthreads();
          // if this thread updates the cmin, set the latest dc
          if (d != std::numeric_limits<coord_t>::max() && d == cmin) {
            *p_dc = j;
          }

          if (threadIdx.x == 0) {
            if (cmin <= *p_cmax) {
              early_break = true;
            }
          }
          __syncthreads();

          left_end = left_begin;
          left_begin = std::max(left_begin - blockDim.x, 0l);
        }

        // right
        if (!early_break && right_begin < right_end) {
          auto d = std::numeric_limits<coord_t>::max();
          unsigned int k;
          for (k = right_begin + threadIdx.x; k < right_end; k += blockDim.x) {
            d = EuclideanDistance2(point_a, v_points_b[k]);
          }
          auto agg_min = BlockReduce(temp_storage).Reduce(d, cub::Min());
          if (threadIdx.x == 0) {
            cmin = std::min(cmin, agg_min);
            atomicAdd(p_compared_pairs, right_end - right_begin);
          }
          __syncthreads();
          if (d != std::numeric_limits<coord_t>::max() && d == cmin) {
            *p_dc = k;
          }

          if (threadIdx.x == 0) {
            if (cmin <= *p_cmax) {
              early_break = true;
            }
          }
          __syncthreads();
          right_begin = right_end;
          right_end =
              std::min((size_t) (right_end + blockDim.x), v_points_b.size());
        }
      }

      __syncthreads();
      if (threadIdx.x == 0 && cmin != std::numeric_limits<coord_t>::max()) {
        atomicMax(p_cmax, cmin);
      }
    }
  });

  LOG(INFO) << "Compared Pairs: " << compared_pairs.get(stream.cuda_stream())
            << " cmax2 " << cmax.get(stream.cuda_stream());

  return sqrt(cmax.get(stream.cuda_stream()));
}

}  // namespace hd
#endif  // HAUSDORFF_DISTANCE_GPU_H
