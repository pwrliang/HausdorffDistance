#ifndef GRID_H
#define GRID_H

#include <optix.h>
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

#include "glog/logging.h"
#include "mbr.h"
#include "utils/array_view.h"
#include "utils/launcher.h"
#include "utils/util.h"

namespace hd {

namespace dev {
namespace details {
template <int N_DIMS>
DEV_HOST_INLINE uint32_t
EncodeCellPos(const typename cuda_vec<unsigned int, N_DIMS>::type& cell_pos,
              uint32_t grid_size) {
  assert(false);
  return -1;
}

template <>
DEV_HOST_INLINE uint32_t EncodeCellPos<2>(const uint2& cell_pos,
                                          uint32_t grid_size) {
  return cell_pos.y * grid_size + cell_pos.x;
}

template <>
DEV_HOST_INLINE uint32_t EncodeCellPos<3>(const uint3& cell_pos,
                                          uint32_t grid_size) {
  return cell_pos.z * grid_size * grid_size + cell_pos.y * grid_size +
         cell_pos.x;
}

template <int N_DIMS>
DEV_HOST_INLINE typename cuda_vec<unsigned int, N_DIMS>::type DecodeCellIdx(
    uint32_t cell_idx, uint32_t grid_size) {
  assert(false);
  return typename cuda_vec<unsigned int, N_DIMS>::type();
}

template <>
DEV_HOST_INLINE uint2 DecodeCellIdx<2>(uint32_t cell_idx, uint32_t grid_size) {
  uint2 cell_pos;
  cell_pos.x = cell_idx % grid_size;
  cell_pos.y = cell_idx / grid_size;
  return cell_pos;
}

template <>
DEV_HOST_INLINE uint3 DecodeCellIdx<3>(uint32_t cell_idx, uint32_t grid_size) {
  uint3 cell_pos;
  cell_pos.z = cell_idx / (grid_size * grid_size);
  cell_idx %= grid_size * grid_size;
  cell_pos.x = cell_idx % grid_size;
  cell_pos.y = cell_idx / grid_size;
  return cell_pos;
}
}

template <typename COORD_T, int N_DIMS>
class Grid {
  using point_t = typename cuda_vec<COORD_T, N_DIMS>::type;
  using mbr_t = Mbr<COORD_T, N_DIMS>;
  using cell_pos_t = typename cuda_vec<uint32_t, N_DIMS>::type;

 public:
  Grid() = default;

  DEV_HOST Grid(const mbr_t& mbr, const ArrayView<mbr_t>& mbrs,
                uint32_t grid_size, const ArrayView<uint32_t>& n_primitives,
                const ArrayView<uint32_t>& row_offset,
                const ArrayView<point_t>& points)
      : mbr_(mbr),
        mbrs_(mbrs),
        grid_size_(grid_size),
        n_primitives_(n_primitives),
        row_offset_(row_offset),
        points_(points) {}

  DEV_INLINE uint32_t CalculateCellIdx(const point_t& p) const {
    auto cell_pos = CalculateCellPos(p);
    return CellPosTo1D(cell_pos);
  }

  DEV_INLINE uint32_t Query(const point_t& p) const {
    auto cell_pos = CalculateCellPos(p);
    auto cell_idx = CellPosTo1D(cell_pos);
    return n_primitives_[cell_idx];
  }

  DEV_INLINE uint32_t begin(uint32_t cell_idx) const {
    return row_offset_[cell_idx];
  }

  DEV_INLINE uint32_t end(uint32_t cell_idx) const {
    return row_offset_[cell_idx + 1];
  }

  DEV_INLINE const point_t& get_point(uint32_t offset) const {
    return points_[offset];
  }

  template <typename U = void,
            typename std::enable_if<N_DIMS == 2, U>::type* = nullptr>
  DEV_INLINE void Insert(const point_t& p) {
    auto cell_p = CalculateCellPos(p);

    auto cell_idx = cell_p.x + cell_p.y * grid_size_;

    atomicAdd(&n_primitives_[cell_idx], 1);
  }

  template <typename U = void,
            typename std::enable_if<N_DIMS == 3, U>::type* = nullptr>
  DEV_INLINE void Insert(const point_t& p) {
    auto cell_p = CalculateCellPos(p);

    auto cell_idx =
        cell_p.x + cell_p.y * grid_size_ + cell_p.y * cell_p.z * grid_size_;

    atomicAdd(&n_primitives_[cell_idx], 1);
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

        atomicAdd(&n_primitives_[cell_idx], 1);
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
          atomicAdd(&n_primitives_[cell_idx], 1);
        }
      }
    }
  }

  DEV_INLINE mbr_t GetCellBounds(const cell_pos_t& cell_pos) const {
    point_t lower_p, upper_p;

    for (int dim = 0; dim < N_DIMS; dim++) {
      auto extent = mbr_.get_extent(dim);
      auto unit = extent / grid_size_;
      auto idx = reinterpret_cast<const unsigned int*>(&cell_pos.x)[dim];
      assert(idx < grid_size_);
      auto begin = unit * idx + mbr_.lower(dim);
      auto end = std::min(begin + unit, mbr_.upper(dim));
      reinterpret_cast<COORD_T*>(&lower_p.x)[dim] = begin;
      reinterpret_cast<COORD_T*>(&upper_p.x)[dim] = end;
    }
    return mbr_t(lower_p, upper_p);
  }

  DEV_INLINE mbr_t GetCellBounds(uint32_t cell_idx) const {
    return GetCellMbr(DecodeCellIdx(cell_idx));
  }

  DEV_HOST_INLINE uint32_t get_grid_size() const { return grid_size_; }

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

  DEV_HOST_INLINE const mbr_t& get_mbr() const { return mbr_; }

  DEV_INLINE const mbr_t& get_mbr(uint32_t cell_idx) const {
    return mbrs_[cell_idx];
  }

  DEV_HOST_INLINE COORD_T get_cell_extent(int dim) const {
    return mbr_.get_extent(dim) / grid_size_;
  }

  DEV_HOST_INLINE uint32_t EncodeCellPos(const cell_pos_t& pos) const {
    return details::EncodeCellPos<N_DIMS>(pos, grid_size_);
  }

  DEV_HOST_INLINE cell_pos_t DecodeCellIdx(uint32_t cell_idx) const {
    return details::DecodeCellIdx<N_DIMS>(cell_idx, grid_size_);
  }

 private:
  mbr_t mbr_;
  ArrayView<mbr_t> mbrs_;
  uint32_t grid_size_;
  ArrayView<uint32_t> n_primitives_;
  ArrayView<uint32_t> row_offset_;
  ArrayView<point_t> points_;

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
  using point_t = typename cuda_vec<COORD_T, N_DIMS>::type;
  using mbr_t = Mbr<COORD_T, N_DIMS>;
  static_assert(N_DIMS == 2 || N_DIMS == 3, "Invalid N_DIMS");

 public:
  Grid() = default;

  explicit Grid(uint32_t max_grid_size) : max_grid_size_(max_grid_size) {
    uint32_t total_cells = 1;
    for (int dim = 0; dim < N_DIMS; dim++) {
      total_cells *= max_grid_size;
    }

    n_primitives_.resize(total_cells);
    mbrs_.resize(total_cells);
  }

  void Clear(const Stream& stream) {
    thrust::fill(thrust::cuda::par.on(stream.cuda_stream()),
                 n_primitives_.begin(), n_primitives_.end(), 0);
    thrust::fill(thrust::cuda::par.on(stream.cuda_stream()), mbrs_.begin(),
                 mbrs_.end(), mbr_t());
    row_offset_.clear();
    points_.clear();
    point_ids_.clear();
  }

  void Init(uint32_t grid_size, const mbr_t& mbr) {
    assert(grid_size <= max_grid_size_);
    grid_size_ = grid_size;
    mbr_ = mbr;
  }

  void Insert(const Stream& stream,
              const thrust::device_vector<point_t>& points,
              bool store_points = false) {
    auto d_grid = DeviceObject();
    auto* p_mbrs = thrust::raw_pointer_cast(mbrs_.data());
    auto* p_n_primitives = thrust::raw_pointer_cast(n_primitives_.data());
    auto n_cells = n_primitives_.size();

    thrust::for_each(thrust::cuda::par.on(stream.cuda_stream()), points.begin(),
                     points.end(), [=] __device__(const point_t& p) mutable {
                       auto cell_idx = d_grid.CalculateCellIdx(p);
                       d_grid.Insert(p);

                       p_mbrs[cell_idx].ExpandAtomic(p);
                     });

    non_empty_cell_ids_.resize(n_cells);
    auto n_non_empty =
        thrust::copy_if(thrust::cuda::par.on(stream.cuda_stream()),
                        thrust::make_counting_iterator<uint32_t>(0),
                        thrust::make_counting_iterator<uint32_t>(n_cells),
                        non_empty_cell_ids_.begin(),
                        [=] __device__(uint32_t cell_idx) {
                          return p_n_primitives[cell_idx] > 0;
                        }) -
        non_empty_cell_ids_.begin();

    non_empty_cell_ids_.resize(n_non_empty);
    non_empty_cell_ids_.shrink_to_fit();

    VLOG(1) << "Non-empty cells: " << n_non_empty
            << " pcnt: " << (float) n_non_empty / n_cells <<
            " points/non-empty cell: " << points.size() / n_non_empty
            << " max points/cell " << *(thrust::max_element(thrust::cuda::par.on(stream.cuda_stream()),
                n_primitives_.begin(), n_primitives_.end()));

    if (store_points) {
      auto n_points = points.size();
      row_offset_.resize(n_cells + 1, 0);
      points_.resize(n_points);
      point_ids_.resize(n_points);

      thrust::inclusive_scan(thrust::cuda::par.on(stream.cuda_stream()),
                             n_primitives_.begin(), n_primitives_.end(),
                             row_offset_.begin() + 1);
      auto* p_row_offset = thrust::raw_pointer_cast(row_offset_.data());
      auto* p_points = thrust::raw_pointer_cast(points_.data());
      auto* p_point_ids = thrust::raw_pointer_cast(point_ids_.data());

      thrust::for_each(thrust::cuda::par.on(stream.cuda_stream()),
                       thrust::make_counting_iterator<uint32_t>(0),
                       thrust::make_counting_iterator<uint32_t>(n_points),
                       [=] __device__(uint32_t id) mutable {
                         const point_t& p = p_points[id];
                         auto cell_idx = d_grid.CalculateCellIdx(p);
                         auto offset = atomicAdd(&p_row_offset[cell_idx], 1);

                         p_points[offset] = p;
                         p_point_ids[offset] = id;
                       });
      // row_offset has been messed up, recompute
      thrust::fill_n(thrust::cuda::par.on(stream.cuda_stream()),
                     row_offset_.begin(), 1, 0);
      thrust::inclusive_scan(thrust::cuda::par.on(stream.cuda_stream()),
                             n_primitives_.begin(), n_primitives_.end(),
                             row_offset_.begin() + 1);
    }
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

  void CellsToTriangles(const Stream& stream,
                        thrust::device_vector<float3>& vertices,
                        thrust::device_vector<uint3>& indices) const {
    auto d_grid = DeviceObject();
    auto n_cells = n_primitives_.size();
    auto* p_n_primitives = thrust::raw_pointer_cast(n_primitives_.data());
    auto* p_non_empty_cell_ids =
        thrust::raw_pointer_cast(non_empty_cell_ids_.data());
    auto n_non_empty = non_empty_cell_ids_.size();

    if (N_DIMS == 2) {
      vertices.resize(n_non_empty * 3);
      indices.resize(n_non_empty);
      auto p_vertices = thrust::raw_pointer_cast(vertices.data());
      auto p_indices = thrust::raw_pointer_cast(indices.data());

      thrust::for_each(thrust::cuda::par.on(stream.cuda_stream()),
                       thrust::make_counting_iterator<uint32_t>(0),
                       thrust::make_counting_iterator<uint32_t>(n_non_empty),
                       [=] __device__(uint32_t i) mutable {
                         auto cell_idx = p_non_empty_cell_ids[i];
                         auto cell_pos = d_grid.DecodeCellIdx(cell_idx);
                         auto cell_bounds = d_grid.GetCellBounds(cell_pos);
                         auto min_x = cell_bounds.lower(0);
                         auto max_x = cell_bounds.upper(0);
                         auto y = cell_bounds.upper(1);

                         float3 p1{min_x, y, 0};
                         float3 p2{max_x, y, -1};
                         float3 p3{min_x, y, 1};

                         p_vertices[i * 3 + 0] = p1;
                         p_vertices[i * 3 + 1] = p2;
                         p_vertices[i * 3 + 2] = p3;
                         p_indices[i] = uint3{i * 3 + 0, i * 3 + 1, i * 3 + 2};
                       });
    }
  }

  dev::Grid<COORD_T, N_DIMS> DeviceObject() const {
    return dev::Grid<COORD_T, N_DIMS>(mbr_, mbrs_, grid_size_, n_primitives_,
                                      row_offset_, points_);
  }

  uint32_t get_grid_size() const { return grid_size_; }

  ArrayView<mbr_t> get_mbrs() const { return {mbrs_}; }

  static uint32_t EstimatedMaxCells(uint32_t memory_quota_mb) {
    return (uint64_t) memory_quota_mb * 1024 * 1024 / sizeof(uint32_t);
  }

  thrust::device_vector<uint32_t>& get_non_empty_cell_ids() {
    return non_empty_cell_ids_;
  }

  const thrust::device_vector<uint32_t>& get_non_empty_cell_ids() const {
    return non_empty_cell_ids_;
  }

 private:
  thrust::device_vector<uint32_t> n_primitives_;
  thrust::device_vector<mbr_t> mbrs_;
  thrust::device_vector<uint32_t> non_empty_cell_ids_;
  thrust::device_vector<uint32_t> row_offset_;
  thrust::device_vector<point_t> points_;
  thrust::device_vector<uint32_t> point_ids_;  // point ids

  mbr_t mbr_;
  uint32_t max_grid_size_;
  uint32_t grid_size_;
};
}  // namespace hd
#endif  // GRID_H
