#ifndef RTSPATIAL_UTILS_QUEUE_H
#define RTSPATIAL_UTILS_QUEUE_H
#include <cooperative_groups.h>

#include "utils/array_view.h"
#include "utils/shared_value.h"
#include "utils/util.h"

namespace hd {
namespace dev {

template <typename T>
class Queue {
 public:
  using value_type = T;

  Queue() = default;

  DEV_HOST explicit Queue(const ArrayView<T>& data, uint32_t* last_pos)
      : data_(data), last_pos_(last_pos) {}

  DEV_INLINE uint32_t Append(const T& item) {
    auto allocation = atomicAdd(last_pos_, 1);
    if (allocation >= data_.size()) {
      printf("out of bound, allocation %u, size %lu\n", allocation,
             data_.size());
    }
    assert(allocation < data_.size());
    data_[allocation] = item;
    return allocation;
  }

  DEV_INLINE uint32_t AppendWarp(const T& item) {
    auto g = cooperative_groups::coalesced_threads();
    uint32_t warp_res;

    if (g.thread_rank() == 0) {
      warp_res = atomicAdd(last_pos_, g.size());
    }
    auto begin = g.shfl(warp_res, 0) + g.thread_rank();
    assert(begin < data_.size());
    data_[begin] = item;
    return begin;
  }

  DEV_INLINE void Clear() const { *last_pos_ = 0; }

  DEV_INLINE T& operator[](uint32_t i) { return data_[i]; }

  DEV_INLINE const T& operator[](uint32_t i) const { return data_[i]; }

  DEV_INLINE uint32_t size() const { return *last_pos_; }

  DEV_INLINE uint32_t capacity() const { return data_.size(); }

  DEV_INLINE void Swap(Queue<T>& rhs) {
    data_.Swap(rhs.data_);
    thrust::swap(last_pos_, rhs.last_pos_);
  }

  DEV_INLINE T* data() { return data_.data(); }

  DEV_INLINE const T* data() const { return data_.data(); }

 private:
  ArrayView<T> data_;
  uint32_t* last_pos_{};
};
}  // namespace dev

template <typename T>
class Queue {
 public:
  using value_type = T;
  using device_t = dev::Queue<T>;

  void Init(uint32_t capacity) {
    data_.resize(capacity);
    counter_.set(0);
  }

  void Clear() { counter_.set(0); }

  void Clear(cudaStream_t stream) { counter_.set(stream, 0); }

  void set_size(size_t n) { counter_.set(n); }

  size_t size() const { return counter_.get(); }

  void set_size(cudaStream_t stream, size_t n) { counter_.set(stream, n); }

  size_t size(cudaStream_t stream) const { return counter_.get(stream); }

  T* data() { return thrust::raw_pointer_cast(data_.data()); }

  const T* data() const { return thrust::raw_pointer_cast(data_.data()); }

  void Append(cudaStream_t stream, const T& item) {
    auto tail = counter_.get(stream);

    CUDA_CHECK(cudaMemcpyAsync(thrust::raw_pointer_cast(data_.data()) + tail,
                               &item, sizeof(T), cudaMemcpyHostToDevice,
                               stream));
    counter_.set(stream, tail + 1);
  }

  void Append(cudaStream_t stream, const Queue& queue) {
    auto tail = counter_.get(stream);
    auto append_size = queue.size(stream);

    thrust::copy(thrust::cuda::par.on(stream), queue.data_.begin(),
                 queue.data_.begin() + append_size, data_.begin() + tail);

    counter_.set(stream, tail + append_size);
  }

  void SetSequence(cudaStream_t stream, uint32_t size, uint32_t start = 0) {
    if (size > data_.size()) {
      data_.resize(size);
    }
    thrust::sequence(thrust::cuda::par.on(stream), data_.begin(),
                     data_.begin() + size, start);
    counter_.set(stream, size);
  }

  template <typename VECTOR_T>
  void CopyTo(VECTOR_T& out, cudaStream_t stream) {
    size_t s = size(stream);
    out = data_;
    out.resize(s);
  }

  device_t DeviceObject() {
    return device_t(ArrayView<T>(data_), counter_.data());
  }

  void Swap(Queue<T>& rhs) {
    data_.swap(rhs.data_);
    counter_.Swap(rhs.counter_);
  }

  size_t capacity() const { return data_.capacity(); }

  thrust::host_vector<T> ToHostVector(const cudaStream_t& stream) const {
    auto n_items = size(stream);
    thrust::host_vector<T> out(n_items);
    thrust::copy(thrust::cuda::par.on(stream), data_.begin(),
                 data_.begin() + n_items, out.begin());
    return out;
  }

 private:
  thrust::device_vector<T> data_;
  SharedValue<uint32_t> counter_{};
};

}  // namespace hd
#endif  // RTSPATIAL_UTILS_QUEUE_H
