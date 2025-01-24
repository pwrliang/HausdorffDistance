#ifndef GRID_H
#define GRID_H

#include <optix.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <utils/array_view.h>
#include <utils/launcher.h>

#include "mbr.h"
#include "utils/launcher.h"
#include "utils/util.h"

namespace hd {

namespace dev {

struct Cell {
  uint32_t n_primitives;
  DEV_HOST Cell() : n_primitives(0) {}
};

template <typename COORD_T, int N_DIMS>
class Grid {
  using cell_t = Cell;
  using point_t = typename cuda_vec<COORD_T, N_DIMS>::type;
  using mbr_t = Mbr<COORD_T, N_DIMS>;

 public:
  Grid() = default;

  DEV_HOST Grid(const mbr_t& mbr, int resolution,
                const ArrayView<cell_t>& cells)
      : mbr_(mbr), resolution_(resolution), cells_(cells) {}

  DEV_INLINE cell_t& Query(const point_t& p) {
    auto cell_pos = CalculateCellPos(p);
    auto cell_idx = CellPosTo1D(cell_pos);
    return cells_[cell_idx];
  }

  DEV_INLINE const cell_t& Query(const point_t& p) const {
    auto cell_pos = CalculateCellPos(p);
    auto cell_idx = CellPosTo1D(cell_pos);
    return cells_[cell_idx];
  }

  template <typename U = void,
            typename std::enable_if<N_DIMS == 2, U>::type* = nullptr>
  DEV_INLINE void Insert(const point_t& p) {
    auto cell_p = CalculateCellPos(p);

    auto cell_idx = cell_p.x + cell_p.y * resolution_;

    atomicAdd(&cells_[cell_idx].n_primitives, 1);
  }

  template <typename U = void,
            typename std::enable_if<N_DIMS == 3, U>::type* = nullptr>
  DEV_INLINE void Insert(const point_t& p) {
    auto cell_p = CalculateCellPos(p);

    auto cell_idx =
        cell_p.x + cell_p.y * resolution_ + cell_p.y * cell_p.z * resolution_;

    atomicAdd(&cells_[cell_idx].n_primitives, 1);
  }

  template <typename U = void,
            typename std::enable_if<N_DIMS == 2, U>::type* = nullptr>
  DEV_INLINE void Insert(const OptixAabb& aabb) {
    auto lower_p = GetLowerPoint(aabb);
    auto upper_p = GetUpperPoint(aabb);
    auto lower_cell_p = CalculateCellPos(lower_p);
    auto upper_cell_p = CalculateCellPos(upper_p);

    for (int i = lower_cell_p.x; i <= upper_cell_p.x; i++) {
      for (int j = lower_cell_p.y; j <= upper_cell_p.y; j++) {
        auto cell_idx = i + j * resolution_;

        atomicAdd(&cells_[cell_idx].n_primitives, 1);
      }
    }
  }

  template <typename U = void,
            typename std::enable_if<N_DIMS == 3, U>::type* = nullptr>
  DEV_INLINE void Insert(const OptixAabb& aabb) {
    auto lower_p = GetLowerPoint(aabb);
    auto upper_p = GetUpperPoint(aabb);
    auto lower_cell_p = CalculateCellPos(lower_p);
    auto upper_cell_p = CalculateCellPos(upper_p);

    for (int i = lower_cell_p.x; i <= upper_cell_p.x; i++) {
      for (int j = lower_cell_p.y; j <= upper_cell_p.y; j++) {
        for (int k = lower_cell_p.z; k <= upper_cell_p.z; k++) {
          auto cell_idx = i + j * resolution_ + k * resolution_ * resolution_;
          atomicAdd(&cells_[cell_idx].n_primitives, 1);
        }
      }
    }
  }

 private:
  mbr_t mbr_;
  int resolution_;
  ArrayView<cell_t> cells_;

  DEV_INLINE typename cuda_vec<unsigned int, N_DIMS>::type CalculateCellPos(
      const point_t& p) const {
    typename cuda_vec<unsigned int, N_DIMS>::type cell_pos;

    for (int dim = 0; dim < N_DIMS; dim++) {
      auto lower = mbr_.lower(dim);
      auto upper = mbr_.upper(dim);
      auto val = reinterpret_cast<const COORD_T*>(&p.x)[dim];

      assert(val >= lower && val <= upper);
      auto norm_val = (val - lower) / (upper - lower);

      reinterpret_cast<unsigned int*>(&cell_pos.x)[dim] =
          std::min(std::max(norm_val * resolution_, (COORD_T) 0.0),
                   resolution_ - (COORD_T) 1.0);
    }
    return cell_pos;
  }

  DEV_INLINE point_t GetLowerPoint(const OptixAabb& aabb) const {
    point_t p;

    for (int dim = 0; dim < N_DIMS; dim++) {
      reinterpret_cast<COORD_T*>(&p.x)[dim] =
          reinterpret_cast<const float*>(&aabb.minX)[dim];
    }
    return p;
  }

  DEV_INLINE point_t GetUpperPoint(const OptixAabb& aabb) const {
    point_t p;

    for (int dim = 0; dim < N_DIMS; dim++) {
      reinterpret_cast<COORD_T*>(&p.x)[dim] =
          reinterpret_cast<const float*>(&aabb.maxX)[dim];
    }
    return p;
  }

  DEV_INLINE uint32_t CellPosTo1D(const uint2& cell_pos) const {
    assert(cell_pos.x < resolution_);
    assert(cell_pos.y < resolution_);
    return cell_pos.x + cell_pos.y * resolution_;
  }

  DEV_INLINE uint32_t CellPosTo1D(const uint3& cell_pos) const {
    assert(cell_pos.x < resolution_);
    assert(cell_pos.y < resolution_);
    assert(cell_pos.z < resolution_);
    return cell_pos.x + cell_pos.y * resolution_ +
           cell_pos.z * resolution_ * resolution_;
  }
};

}  // namespace dev

template <typename COORD_T, int N_DIMS>
class Grid {
  using cell_t = dev::Cell;
  using point_t = typename cuda_vec<COORD_T, N_DIMS>::type;
  using mbr_t = Mbr<COORD_T, N_DIMS>;

 public:
  Grid() = default;

  explicit Grid(int resolution) : resolution_(resolution) {
    uint32_t total_cells = 1;
    for (int dim = 0; dim < N_DIMS; dim++) {
      total_cells *= resolution_;
    }

    cells_.resize(total_cells);
  }

  void Clear(cudaStream_t cuda_stream) {
    cell_t cell;
    thrust::fill(thrust::cuda::par.on(cuda_stream), cells_.begin(),
                 cells_.end(), cell);
  }

  void set_mbr(const mbr_t& mbr) { mbr_ = mbr; }

  void Insert(cudaStream_t cuda_stream,
              const thrust::device_vector<point_t>& points) {
    auto d_grid = DeviceObject();
    thrust::for_each(
        thrust::cuda::par.on(cuda_stream), points.begin(), points.end(),
        [=] __device__(const point_t& p) mutable { d_grid.Insert(p); });
  }

  void Insert(cudaStream_t cuda_stream,
              const thrust::device_vector<OptixAabb>& aabbs) {
    auto d_grid = DeviceObject();
    thrust::for_each(
        thrust::cuda::par.on(cuda_stream), aabbs.begin(), aabbs.end(),
        [=] __device__(const OptixAabb& aabb) mutable { d_grid.Insert(aabb); });
  }

  dev::Grid<COORD_T, N_DIMS> DeviceObject() {
    return dev::Grid<COORD_T, N_DIMS>(mbr_, resolution_, cells_);
  }
  const dev::Grid<COORD_T, N_DIMS> DeviceObject() const {
    return dev::Grid<COORD_T, N_DIMS>(mbr_, resolution_, cells_);
  }

 private:
  thrust::device_vector<cell_t> cells_;
  mbr_t mbr_;
  int resolution_;
};
}  // namespace hd
#endif  // GRID_H
