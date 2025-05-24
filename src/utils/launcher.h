#ifndef RAYJOIN_LAUNCHER_H
#define RAYJOIN_LAUNCHER_H
#include "utils/stream.h"
#include "utils/util.h"
namespace hd {
template <typename F, typename... Args>
__global__ void KernelWrapper(F f, Args... args) {
  f(args...);
}

template <typename F, typename... Args>
__global__ void KernelWrapperForEach(size_t size, F f, Args... args) {
  for (size_t i = TID_1D; i < size; i += TOTAL_THREADS_1D) {
    f(i, args...);
  }
}

template <typename F, typename... Args>
void LaunchKernel(const Stream& stream, dim3 grid_size, dim3 block_size, F f,
                  Args&&... args) {
  KernelWrapper<<<grid_size, block_size, 0, stream.cuda_stream()>>>(
      f, std::forward<Args>(args)...);
}

template <typename F, typename... Args>
int2 GetKernelLaunchParams(F f, Args&&... args) {
  int grid_size, block_size;

  CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(
      &grid_size, &block_size, KernelWrapper<F, Args...>, 0,
      reinterpret_cast<int>(MAX_BLOCK_SIZE)));
  int numBlocksPerSm;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &numBlocksPerSm, KernelWrapper<F, Args...>, block_size, 0));

  return {numBlocksPerSm, block_size};
}

template <typename F, typename... Args>
void LaunchKernel(const Stream& stream, F f, Args&&... args) {
  int grid_size, block_size;

  CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(
      &grid_size, &block_size, KernelWrapper<F, Args...>, 0,
      reinterpret_cast<int>(MAX_BLOCK_SIZE)));

  KernelWrapper<<<grid_size, block_size, 0, stream.cuda_stream()>>>(
      f, std::forward<Args>(args)...);
}

template <typename F, typename... Args>
void LaunchCooperativeKernel(const Stream& stream, F f, Args&&... args) {
  int grid_size, block_size;

  CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(
      &grid_size, &block_size, KernelWrapper<F, Args...>, 0,
      reinterpret_cast<int>(MAX_BLOCK_SIZE)));

  dim3 grid_dims{(uint32_t) grid_size, 1, 1},
      block_dims{(uint32_t) block_size, 1, 1};

  void* kernelArgs[] = {&f, std::forward<Args>(args)...};

  cudaLaunchCooperativeKernel((void*) KernelWrapper<F, Args...>, grid_dims,
                              block_dims, kernelArgs, 0, stream.cuda_stream());
}

template <typename F, typename... Args>
void LaunchKernel(const Stream& stream, size_t size, F f, Args&&... args) {
  int grid_size, block_size;

  KernelSizing(grid_size, block_size, size);
  KernelWrapper<<<grid_size, block_size, 0, stream.cuda_stream()>>>(
      f, std::forward<Args>(args)...);
}

template <typename F, typename... Args>
void ForEach(const Stream& stream, size_t size, F f, Args&&... args) {
  int grid_size, block_size;

  KernelSizing(grid_size, block_size, size);
  KernelWrapperForEach<<<grid_size, block_size, 0, stream.cuda_stream()>>>(
      size, f, std::forward<Args>(args)...);
}

template <typename F, typename... Args>
void LaunchKernelFix(const Stream& stream, size_t size, F f, Args&&... args) {
  KernelWrapper<<<256, 256, 0, stream.cuda_stream()>>>(
      f, std::forward<Args>(args)...);
}
}  // namespace hd
#endif  // RAYJOIN_LAUNCHER_H
