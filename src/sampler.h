#ifndef RTSPATIAL_DETAILS_SAMPLER_H
#define RTSPATIAL_DETAILS_SAMPLER_H
#include <curand_kernel.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

#include "utils/array_view.h"

namespace hd {
class Sampler {
 public:
  Sampler() : seed_(0) {}

  explicit Sampler(int seed) : seed_(seed) {}

  void Init(size_t max_samples) {
    int seed = seed_;

    if (max_samples > states_.size()) {
      states_.resize(max_samples);
      auto* states = thrust::raw_pointer_cast(states_.data());

      thrust::for_each(
          thrust::device, thrust::make_counting_iterator<uint32_t>(0),
          thrust::make_counting_iterator<uint32_t>(max_samples),
          [=] __device__(uint32_t i) { curand_init(seed, i, 0, &states[i]); });
    }
  }

  template <typename T>
  thrust::device_vector<T> Sample(cudaStream_t cuda_stream, ArrayView<T> array,
                                  uint32_t sample_size) {
    auto n = array.size();
    sample_size = std::min(sample_size, (uint32_t) states_.size());
    thrust::device_vector<T> samples(sample_size);
    auto* states = thrust::raw_pointer_cast(states_.data());
    auto* p_samples = thrust::raw_pointer_cast(samples.data());

    thrust::for_each(thrust::cuda::par.on(cuda_stream),
                     thrust::make_counting_iterator<uint32_t>(0),
                     thrust::make_counting_iterator<uint32_t>(sample_size),
                     [=] __device__(uint32_t i) {
                       uint32_t index = curand(&states[i]) % n;

                       p_samples[i] = array[index];
                     });
    return samples;
  }

  thrust::device_vector<uint32_t> Sample(cudaStream_t cuda_stream,
                                         uint32_t max_id,
                                         uint32_t sample_size) {
    sample_size = std::min(sample_size, (uint32_t) states_.size());
    thrust::device_vector<uint32_t> samples(sample_size);
    auto* states = thrust::raw_pointer_cast(states_.data());
    auto* p_samples = thrust::raw_pointer_cast(samples.data());

    thrust::for_each(thrust::cuda::par.on(cuda_stream),
                     thrust::make_counting_iterator<uint32_t>(0),
                     thrust::make_counting_iterator<uint32_t>(sample_size),
                     [=] __device__(uint32_t i) {
                       p_samples[i] = curand(&states[i]) % max_id;
                     });
    return samples;
  }

 private:
  int seed_;
  thrust::device_vector<curandState> states_;
};
}  // namespace hd
#endif  // RTSPATIAL_DETAILS_SAMPLER_H
