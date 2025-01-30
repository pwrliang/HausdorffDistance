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
      : mbr_(mbr), grid_size_(resolution), cells_(cells) {}

  DEV_INLINE const cell_t& Query(const point_t& p) const {
    auto cell_pos = CalculateCellPos(p);
    auto cell_idx = CellPosTo1D(cell_pos);
    return cells_[cell_idx];
  }

  template <typename U = void,
            typename std::enable_if<N_DIMS == 2, U>::type* = nullptr>
  DEV_INLINE void Insert(const point_t& p) {
    auto cell_p = CalculateCellPos(p);

    auto cell_idx = cell_p.x + cell_p.y * grid_size_;

    atomicAdd(&cells_[cell_idx].n_primitives, 1);
  }

  template <typename U = void,
            typename std::enable_if<N_DIMS == 3, U>::type* = nullptr>
  DEV_INLINE void Insert(const point_t& p) {
    auto cell_p = CalculateCellPos(p);

    auto cell_idx =
        cell_p.x + cell_p.y * grid_size_ + cell_p.y * cell_p.z * grid_size_;

    atomicAdd(&cells_[cell_idx].n_primitives, 1);
  }

  template <typename U = void,
            typename std::enable_if<N_DIMS == 2, U>::type* = nullptr>
  DEV_INLINE void Insert(const OptixAabb& aabb) {
    auto lower_p = GetLowerPoint(aabb);
    auto upper_p = GetUpperPoint(aabb);
    auto lower_cell_p = CalculateCellPos(lower_p);
    auto upper_cell_p = CalculateCellPos(upper_p);

    for (int j = lower_cell_p.y; j <= upper_cell_p.y; j++) {
      for (int i = lower_cell_p.x; i <= upper_cell_p.x; i++) {
        auto cell_idx = i + j * grid_size_;

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

    for (int k = lower_cell_p.z; k <= upper_cell_p.z; k++) {
      for (int j = lower_cell_p.y; j <= upper_cell_p.y; j++) {
        for (int i = lower_cell_p.x; i <= upper_cell_p.x; i++) {
          auto cell_idx = i + j * grid_size_ + k * grid_size_ * grid_size_;
          atomicAdd(&cells_[cell_idx].n_primitives, 1);
        }
      }
    }
  }

 private:
  mbr_t mbr_;
  int grid_size_;
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
          std::min(std::max(norm_val * grid_size_, (COORD_T) 0.0),
                   grid_size_ - (COORD_T) 1.0);
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
    assert(cell_pos.x < grid_size_);
    assert(cell_pos.y < grid_size_);
    return cell_pos.x + cell_pos.y * grid_size_;
  }

  DEV_INLINE uint32_t CellPosTo1D(const uint3& cell_pos) const {
    assert(cell_pos.x < grid_size_);
    assert(cell_pos.y < grid_size_);
    assert(cell_pos.z < grid_size_);
    return cell_pos.x + cell_pos.y * grid_size_ +
           cell_pos.z * grid_size_ * grid_size_;
  }
};

}  // namespace dev

template <typename COORD_T, int N_DIMS>
class Grid {
  using cell_t = dev::Cell;
  using point_t = typename cuda_vec<COORD_T, N_DIMS>::type;
  using mbr_t = Mbr<COORD_T, N_DIMS>;
  static_assert(N_DIMS == 2 || N_DIMS == 3, "Invalid N_DIMS");

 public:
  Grid() = default;

  explicit Grid(int max_grid_size) : max_grid_size_(max_grid_size) {
    uint32_t total_cells = 1;
    for (int dim = 0; dim < N_DIMS; dim++) {
      total_cells *= max_grid_size;
    }

    cells_.resize(total_cells);
  }

  void Clear(const Stream& stream) {
    cell_t cell;
    thrust::fill(thrust::cuda::par.on(stream.cuda_stream()), cells_.begin(),
                 cells_.end(), cell);
  }

  void Init(int grid_size, const mbr_t& mbr) {
    assert(grid_size <= max_grid_size_);
    grid_size_ = grid_size;
    mbr_ = mbr;
  }

  void Insert(const Stream& stream,
              const thrust::device_vector<point_t>& points) {
    auto d_grid = DeviceObject();
    thrust::for_each(thrust::cuda::par.on(stream.cuda_stream()), points.begin(),
                     points.end(), [=] __device__(const point_t& p) mutable {
                       d_grid.Insert(p);
                     });
  }

  void Insert(const Stream& stream,
              const thrust::device_vector<OptixAabb>& aabbs) {
    ArrayView<OptixAabb> v_aabbs(aabbs);
    auto d_grid = DeviceObject();

    LaunchKernel(stream, [=] __device__() mutable {
      for (auto aabb_id = TID_1D; aabb_id < v_aabbs.size();
           aabb_id += TOTAL_THREADS_1D) {
        const auto& aabb = v_aabbs[aabb_id];

        d_grid.Insert(aabb);
      }
    });
  }

  dev::Grid<COORD_T, N_DIMS> DeviceObject() const {
    return dev::Grid<COORD_T, N_DIMS>(mbr_, grid_size_, cells_);
  }

  int get_grid_size() const { return grid_size_; }

 private:
  thrust::device_vector<cell_t> cells_;
  mbr_t mbr_;
  int max_grid_size_;
  int grid_size_;
};
}  // namespace hd
#endif  // GRID_H
