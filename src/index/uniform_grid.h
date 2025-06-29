#ifndef HAUSDORFF_DISTANCE_INDEX_UNIFORM_GRID_H
#define HAUSDORFF_DISTANCE_INDEX_UNIFORM_GRID_H

#include <glog/logging.h>
#include <hdr/hdr_histogram.h>
#include <optix.h>
#include <thrust/binary_search.h>
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>

#include <iostream>
#include <nlohmann/json.hpp>

#include "geoms/mbr.h"
#include "running_stats.h"
#include "utils/array_view.h"
#include "utils/bitset.h"
#include "utils/launcher.h"
#include "utils/queue.h"
#include "utils/stopwatch.h"
#include "utils/util.h"

namespace hd {

namespace dev {
namespace details {
template <int N_DIMS>
DEV_HOST_INLINE uint64_t
EncodeCellPos(const typename cuda_vec<unsigned int, N_DIMS>::type& cell_pos,
              const typename cuda_vec<unsigned int, N_DIMS>::type& grid_size) {
  assert(false);
  return -1;
}

template <>
DEV_HOST_INLINE uint64_t EncodeCellPos<2>(const uint2& cell_pos,
                                          const uint2& grid_size) {
  return (uint64_t) cell_pos.y * grid_size.x + cell_pos.x;
}

template <>
DEV_HOST_INLINE uint64_t EncodeCellPos<3>(const uint3& cell_pos,
                                          const uint3& grid_size) {
  return (uint64_t) cell_pos.z * grid_size.x * grid_size.y +
         (uint64_t) cell_pos.y * grid_size.x + cell_pos.x;
}

template <int N_DIMS>
DEV_HOST_INLINE typename cuda_vec<unsigned int, N_DIMS>::type DecodeCellIdx(
    uint64_t cell_idx,
    const typename cuda_vec<unsigned int, N_DIMS>::type& grid_size) {
  assert(false);
  return typename cuda_vec<unsigned int, N_DIMS>::type();
}

template <>
DEV_HOST_INLINE uint2 DecodeCellIdx<2>(uint64_t cell_idx,
                                       const uint2& grid_size) {
  uint2 cell_pos;
  cell_pos.x = cell_idx % grid_size.x;
  cell_pos.y = cell_idx / grid_size.x;
  return cell_pos;
}

template <>
DEV_HOST_INLINE uint3 DecodeCellIdx<3>(uint64_t cell_idx,
                                       const uint3& grid_size) {
  uint3 cell_pos;
  cell_pos.z = cell_idx / ((uint64_t) grid_size.x * grid_size.y);
  cell_idx %= (uint64_t) grid_size.x * grid_size.y;
  cell_pos.x = cell_idx % grid_size.x;
  cell_pos.y = cell_idx / grid_size.x;
  return cell_pos;
}

}  // namespace details

template <typename COORD_T, int N_DIMS>
class UniformGrid {
  using point_t = typename cuda_vec<COORD_T, N_DIMS>::type;
  using mbr_t = Mbr<COORD_T, N_DIMS>;
  using cell_idx_t = typename cuda_vec<uint32_t, N_DIMS>::type;

 public:
  UniformGrid() = default;

  DEV_HOST UniformGrid(const mbr_t& mbr, const cell_idx_t& grid_size)
      : mbr_(mbr), grid_size_(grid_size) {}

  DEV_HOST UniformGrid(const mbr_t& mbr, const cell_idx_t& grid_size,
                       const ArrayView<uint64_t>& cell_ids)
      : mbr_(mbr), grid_size_(grid_size), cell_ids_(cell_ids) {}

  DEV_HOST UniformGrid(const mbr_t& mbr, const cell_idx_t& grid_size,
                       const ArrayView<uint64_t>& cell_ids,
                       const ArrayView<uint32_t>& prefix_sum,
                       const ArrayView<uint32_t> point_ids)
      : mbr_(mbr),
        grid_size_(grid_size),
        cell_ids_(cell_ids),
        prefix_sum_(prefix_sum),
        point_ids_(point_ids) {}

  DEV_INLINE mbr_t GetCellBounds(const cell_idx_t& cell_pos) const {
    point_t lower_p, upper_p;

    for (int dim = 0; dim < N_DIMS; dim++) {
      auto extent = mbr_.get_extent(dim);
      auto grid_size = get_grid_size(dim);
      auto unit = extent / grid_size;
      auto idx = reinterpret_cast<const unsigned int*>(&cell_pos.x)[dim];
      assert(idx < grid_size);
      // Calculate a conservative bounds
      auto unit_lower = nextafter(unit, unit - 1);
      unit_lower = nextafter(unit_lower, unit_lower - 1);
      auto unit_upper = nextafter(unit, unit + 1);
      unit_upper = nextafter(unit_upper, unit_upper + 1);
      auto begin = idx * unit_lower + mbr_.lower(dim);
      auto end =
          std::min(mbr_.lower(dim) + (idx + 1) * unit_upper, mbr_.upper(dim));

      reinterpret_cast<COORD_T*>(&lower_p.x)[dim] = begin;
      reinterpret_cast<COORD_T*>(&upper_p.x)[dim] = end;
    }
    return mbr_t(lower_p, upper_p);
  }

  DEV_INLINE mbr_t GetCellBounds(uint64_t cell_idx) const {
    return GetCellBounds(DecodeCellIdx(cell_idx));
  }

  DEV_HOST_INLINE unsigned int get_grid_size(int dim) const {
    return reinterpret_cast<const unsigned int*>(&grid_size_.x)[dim];
  }

  DEV_INLINE typename cuda_vec<unsigned int, N_DIMS>::type CalculateCellPos(
      const point_t& p) const {
    typename cuda_vec<unsigned int, N_DIMS>::type cell_pos;

    for (int dim = 0; dim < N_DIMS; dim++) {
      auto lower = mbr_.lower(dim);
      auto upper = mbr_.upper(dim);
      auto grid_size = get_grid_size(dim);
      auto val = reinterpret_cast<const COORD_T*>(&p.x)[dim];

      assert(val >= lower && val <= upper);
      auto norm_val = (val - lower) / (upper - lower);

      assert(norm_val >= 0);
      assert(norm_val <= 1);

      reinterpret_cast<unsigned int*>(&cell_pos.x)[dim] =
          std::min(std::max(norm_val * grid_size, (COORD_T) 0.0),
                   grid_size - (COORD_T) 1.0);
    }
    return cell_pos;
  }

  DEV_HOST_INLINE const mbr_t& get_mbr() const { return mbr_; }

  DEV_HOST_INLINE uint64_t EncodeCellPos(const cell_idx_t& pos) const {
    return details::EncodeCellPos<N_DIMS>(pos, grid_size_);
  }

  DEV_HOST_INLINE cell_idx_t DecodeCellIdx(uint64_t cell_idx) const {
    return details::DecodeCellIdx<N_DIMS>(cell_idx, grid_size_);
  }

  DEV_HOST_INLINE bool IsOverlapping(const point_t& p) const {
    auto cell_pos = CalculateCellPos(p);
    auto cell_idx = EncodeCellPos(cell_pos);
    auto it = thrust::lower_bound(thrust::seq, cell_ids_.begin(),
                                  cell_ids_.end(), cell_idx);
    return it != cell_ids_.end();
  }

  DEV_HOST_INLINE uint32_t GetNonEmptyCellId(const point_t& p) const {
    auto cell_pos = CalculateCellPos(p);
    auto cell_idx = EncodeCellPos(cell_pos);
    auto it = thrust::lower_bound(thrust::seq, cell_ids_.begin(),
                                  cell_ids_.end(), cell_idx);
    auto cell_renumbered_idx = it - cell_ids_.begin();
    assert(cell_renumbered_idx >= 0 && cell_renumbered_idx < cell_ids_.size());
    return cell_renumbered_idx;
  }

  DEV_INLINE auto begin(uint32_t cell_idx) const {
    return prefix_sum_[cell_idx];
  }

  DEV_INLINE auto end(uint32_t cell_idx) const {
    return prefix_sum_[cell_idx + 1];
  }

  DEV_INLINE uint32_t get_point_id(uint32_t offset) const {
    return point_ids_[offset];
  }

 private:
  mbr_t mbr_;
  cell_idx_t grid_size_;
  ArrayView<uint64_t> cell_ids_;
  ArrayView<uint32_t> prefix_sum_;
  ArrayView<uint32_t> point_ids_;
};

}  // namespace dev

template <typename COORD_T, int N_DIMS>
class UniformGrid {
  using point_t = typename cuda_vec<COORD_T, N_DIMS>::type;
  using mbr_t = Mbr<COORD_T, N_DIMS>;
  using cell_idx_t = typename cuda_vec<uint32_t, N_DIMS>::type;

  static_assert(N_DIMS == 2 || N_DIMS == 3, "Invalid N_DIMS");

 public:
  UniformGrid() = default;

  static typename cuda_vec<unsigned int, N_DIMS>::type CalculateGridResolution(
      const mbr_t& mbr, unsigned int n_points, int n_points_per_cell) {
    double volume = 1;
    COORD_T extents[N_DIMS];

    for (int dim = 0; dim < N_DIMS; dim++) {
      volume *= mbr.get_extent(dim);
    }
    CHECK_GT(volume, 0);
    double s = 0;

    if (N_DIMS == 2) {
      s = sqrt(volume * n_points_per_cell / n_points);
    } else if (N_DIMS == 3) {
      s = cbrt(volume * n_points_per_cell / n_points);
    }
    typename cuda_vec<unsigned int, N_DIMS>::type dims;
    for (int dim = 0; dim < N_DIMS; dim++) {
      reinterpret_cast<unsigned int*>(&dims.x)[dim] =
          ceil(mbr.get_extent(dim) / s);
    }
    return dims;
  }

  void Clear(const Stream& stream) {
    occupied_cells_.Clear(stream.cuda_stream());
    cell_ids_.clear();
    point_ids_.clear();
  }

  void Init(const cell_idx_t& grid_size, const mbr_t& mbr) {
    grid_size_ = grid_size;
    mbr_ = mbr;
    occupied_cells_.Init(get_grid_size());
  }

  uint64_t get_grid_size() const {
    uint64_t total_cells = 1;
    for (int dim = 0; dim < N_DIMS; dim++) {
      total_cells *= reinterpret_cast<const unsigned int*>(&grid_size_.x)[dim];
    }
    return total_cells;
  }

  void AutotuneGridSize(const Stream& stream, const mbr_t& mbr,
                        const thrust::device_vector<point_t>& query_points,
                        const thrust::device_vector<point_t>& points) {
    thrust::device_vector<uint32_t> n_primitives;
    SharedValue<uint32_t> tmp_offset;
    auto min_cost = std::numeric_limits<uint64_t>::max();

    for (int n_points_per_cell = 1; n_points_per_cell < 20;
         n_points_per_cell++) {
      auto grid_size =
          CalculateGridResolution(mbr, points.size(), n_points_per_cell);
      dev::UniformGrid<COORD_T, N_DIMS> d_grid(mbr, grid_size);
      uint64_t total_cells = 1;
      for (int dim = 0; dim < N_DIMS; dim++) {
        total_cells *= reinterpret_cast<const unsigned int*>(&grid_size.x)[dim];
      }
      occupied_cells_.Init(total_cells);
      occupied_cells_.Clear(stream.cuda_stream());
      tmp_offset.set(stream.cuda_stream(), 0);

      auto d_occupied_cells = occupied_cells_.DeviceObject();
      thrust::for_each(thrust::cuda::par.on(stream.cuda_stream()),
                       points.begin(), points.end(),
                       [=] __device__(const point_t& p) mutable {
                         auto cell_p = d_grid.CalculateCellPos(p);
                         auto cell_idx = d_grid.EncodeCellPos(cell_p);
                         auto cell_mbr = d_grid.GetCellBounds(cell_idx);
                         assert(cell_mbr.Contains(p));
                         d_occupied_cells.set_bit_atomic(cell_idx);
                       });
      auto non_empty_cells =
          occupied_cells_.GetPositiveCount(stream.cuda_stream());

      cell_ids_.resize(non_empty_cells);
      n_primitives.resize(non_empty_cells, 0);
      prefix_sum_.resize(non_empty_cells + 1, 0);

      auto* p_n_primitives = thrust::raw_pointer_cast(n_primitives.data());

      // collect non-empty cell ids
      thrust::copy_if(
          thrust::cuda::par.on(stream.cuda_stream()),
          thrust::make_counting_iterator<size_t>(0),
          thrust::make_counting_iterator<size_t>(occupied_cells_.GetSize()),
          cell_ids_.begin(), [=] __device__(size_t cell_idx) {
            return d_occupied_cells.get_bit(cell_idx);
          });

      // sort cell ids for binary search
      thrust::sort(thrust::cuda::par.on(stream.cuda_stream()),
                   cell_ids_.begin(), cell_ids_.end());
      d_grid = dev::UniformGrid<COORD_T, N_DIMS>(mbr, grid_size, cell_ids_);
      // count the number of primitives per cell
      thrust::for_each(thrust::cuda::par.on(stream.cuda_stream()),
                       points.begin(), points.end(),
                       [=] __device__(const point_t& p) mutable {
                         auto cell_renumbered_idx = d_grid.GetNonEmptyCellId(p);
                         atomicAdd(&p_n_primitives[cell_renumbered_idx], 1);
                       });

      auto cost = thrust::transform_reduce(
          thrust::cuda::par.on(stream.cuda_stream()), query_points.begin(),
          query_points.end(),
          [=] __device__(const point_t& p) -> uint32_t {
            if (mbr.Contains(p)) {
              return p_n_primitives[d_grid.GetNonEmptyCellId(p)];
            }
            return 0;
          },
          0ul, thrust::plus<uint64_t>());

      LOG(INFO) << "Cost: " << cost << " points/cell " << n_points_per_cell;
      min_cost = std::min(min_cost, cost);
    }
    LOG(INFO) << "Minimum cost " << min_cost;
  }

  void Insert(const Stream& stream,
              const thrust::device_vector<point_t>& points,
              bool insert_point_ids = true) {
    Stopwatch sw;
    sw.start();
    // auto d_grid = DeviceObject();
    dev::UniformGrid<COORD_T, N_DIMS> d_grid(mbr_, grid_size_);
    thrust::device_vector<uint32_t> n_primitives;
    SharedValue<uint32_t> tmp_offset;
    auto d_occupied_cells = occupied_cells_.DeviceObject();

    occupied_cells_.Clear(stream.cuda_stream());
    tmp_offset.set(stream.cuda_stream(), 0);

    thrust::for_each(thrust::cuda::par.on(stream.cuda_stream()), points.begin(),
                     points.end(), [=] __device__(const point_t& p) mutable {
                       auto cell_p = d_grid.CalculateCellPos(p);
                       auto cell_idx = d_grid.EncodeCellPos(cell_p);
                       auto cell_mbr = d_grid.GetCellBounds(cell_idx);
                       assert(cell_mbr.Contains(p));
                       d_occupied_cells.set_bit_atomic(cell_idx);
                     });

    auto non_empty_cells =
        occupied_cells_.GetPositiveCount(stream.cuda_stream());

    stats_["TotalCells"] = get_grid_size();
    stats_["NonEmptyCells"] = non_empty_cells/get_grid_size();

    cell_ids_.resize(non_empty_cells);
    n_primitives.resize(non_empty_cells, 0);
    prefix_sum_.resize(non_empty_cells + 1, 0);

    auto* p_cell_ids = thrust::raw_pointer_cast(cell_ids_.data());
    auto* p_n_primitives = thrust::raw_pointer_cast(n_primitives.data());
    auto* p_offset = tmp_offset.data();

    // collect non-empty cell ids
    thrust::copy_if(
        thrust::cuda::par.on(stream.cuda_stream()),
        thrust::make_counting_iterator<size_t>(0),
        thrust::make_counting_iterator<size_t>(occupied_cells_.GetSize()),
        cell_ids_.begin(), [=] __device__(size_t cell_idx) {
          return d_occupied_cells.get_bit(cell_idx);
        });

    // sort cell ids for binary search
    thrust::sort(thrust::cuda::par.on(stream.cuda_stream()), cell_ids_.begin(),
                 cell_ids_.end());
    d_grid = dev::UniformGrid<COORD_T, N_DIMS>(mbr_, grid_size_, cell_ids_);
    // count the number of primitives per cell
    thrust::for_each(thrust::cuda::par.on(stream.cuda_stream()), points.begin(),
                     points.end(), [=] __device__(const point_t& p) mutable {
                       auto cell_renumbered_idx = d_grid.GetNonEmptyCellId(p);
                       atomicAdd(&p_n_primitives[cell_renumbered_idx], 1);
                     });

    stats_["MaxPoints"] = stats_["MaxPoints"] = thrust::reduce(
        thrust::cuda::par.on(stream.cuda_stream()), n_primitives.begin(),
        n_primitives.end(), 0u, thrust::maximum<uint32_t>());
    stats_["GiniIndex"] = gini_index_thrust(stream, n_primitives);

    thrust::inclusive_scan(thrust::cuda::par.on(stream.cuda_stream()),
                           n_primitives.begin(), n_primitives.end(),
                           prefix_sum_.begin() + 1);

    if (insert_point_ids) {
      // write back points
      point_ids_.resize(points.size());
      auto* p_prefix_sum = thrust::raw_pointer_cast(prefix_sum_.data());
      auto* p_point_ids = thrust::raw_pointer_cast(point_ids_.data());

      // fill point ids to the corresponding cells
      thrust::for_each(
          thrust::cuda::par.on(stream.cuda_stream()),
          thrust::make_zip_iterator(thrust::make_tuple(
              thrust::make_counting_iterator<uint32_t>(0), points.begin())),
          thrust::make_zip_iterator(thrust::make_tuple(
              thrust::make_counting_iterator<uint32_t>(points.size()),
              points.end())),
          [=] __device__(
              const thrust::tuple<uint32_t, point_t>& tuple) mutable {
            auto point_idx = thrust::get<0>(tuple);
            const auto& p = thrust::get<1>(tuple);
            auto cell_pos = d_grid.CalculateCellPos(p);
            auto cell_idx = d_grid.EncodeCellPos(cell_pos);
            auto cell_renumbered_idx =
                thrust::lower_bound(thrust::seq, p_cell_ids,
                                    p_cell_ids + non_empty_cells, cell_idx) -
                p_cell_ids;
            assert(cell_renumbered_idx < non_empty_cells);
            assert(p_cell_ids[cell_renumbered_idx] == cell_idx);
            auto last = atomicAdd(&p_prefix_sum[cell_renumbered_idx], 1);

            p_point_ids[last] = point_idx;
          });

      thrust::fill_n(thrust::cuda::par.on(stream.cuda_stream()),
                     prefix_sum_.begin(), 1, 0);
      // restore prefix sum
      thrust::inclusive_scan(thrust::cuda::par.on(stream.cuda_stream()),
                             n_primitives.begin(), n_primitives.end(),
                             prefix_sum_.begin() + 1);
#ifndef NDEBUG
      auto* p_points = thrust::raw_pointer_cast(points.data());
      thrust::for_each(
          thrust::cuda::par.on(stream.cuda_stream()),
          thrust::make_counting_iterator<uint32_t>(0),
          thrust::make_counting_iterator<uint32_t>(non_empty_cells),
          [=] __device__(uint32_t cell_idx) {
            auto begin = p_prefix_sum[cell_idx];
            auto end = p_prefix_sum[cell_idx + 1];
            assert(begin < end);

            uint64_t original_cell_idx = p_cell_ids[cell_idx];
            auto cell_mbr = d_grid.GetCellBounds(original_cell_idx);

            for (int i = begin; i < end; i++) {
              auto point_idx = p_point_ids[i];
              const auto& p = p_points[point_idx];

              assert(cell_mbr.Contains(p));
            }
          });
#endif
    }
    sw.stop();
    stats_["BuildTime"] = sw.ms();
    auto json_dims = nlohmann::json::array();
    for (int dim = 0; dim < N_DIMS; dim++) {
      json_dims.push_back(reinterpret_cast<unsigned int*>(&grid_size_)[dim]);
    }
    stats_["GridSize"] = json_dims;
    stats_["CellDiagonalLength"] = GetCellDigonalLength();

    // now, n_primitves is useless
    thrust::sort(thrust::cuda::par.on(stream.cuda_stream()),
                 n_primitives.begin(), n_primitives.end());
    uint32_t median_points_per_cell = n_primitives[n_primitives.size() / 2];

    stats_["MedianPointsPerCell"] = median_points_per_cell;
  }

  dev::UniformGrid<COORD_T, N_DIMS> DeviceObject() {
    return dev::UniformGrid<COORD_T, N_DIMS>(mbr_, grid_size_, cell_ids_,
                                             prefix_sum_, point_ids_);
  }

  static uint32_t EstimatedMaxCells(uint32_t memory_quota_mb) {
    return (uint64_t) memory_quota_mb * 1024 * 1024 / sizeof(uint32_t);
  }

  thrust::device_vector<mbr_t> GetCellMbrs(const Stream& stream) {
    thrust::device_vector<mbr_t> mbrs(cell_ids_.size());
    dev::UniformGrid<COORD_T, N_DIMS> d_grid(mbr_, grid_size_);

    thrust::transform(thrust::cuda::par.on(stream.cuda_stream()),
                      cell_ids_.begin(), cell_ids_.end(), mbrs.begin(),
                      [=] __device__(uint64_t cell_id) {
                        return d_grid.GetCellBounds(cell_id);
                      });
    return mbrs;
  }

  thrust::device_vector<point_t> GetCellCenters(const Stream& stream) {
    thrust::device_vector<point_t> centers(cell_ids_.size());
    auto d_grid = DeviceObject();

    thrust::transform(thrust::cuda::par.on(stream.cuda_stream()),
                      cell_ids_.begin(), cell_ids_.end(), centers.begin(),
                      [=] __device__(uint64_t cell_id) {
                        return d_grid.GetCellBounds(cell_id).get_center();
                      });
    return centers;
  }

  void FilterOverlapPoints(const Stream& stream,
                           const thrust::device_vector<point_t>& points,
                           Queue<uint32_t>& overlapping,
                           Queue<uint32_t>& no_overlapping) {
    auto d_grid = DeviceObject();
    auto d_overlapping = overlapping.DeviceObject();
    auto d_no_overlapping = no_overlapping.DeviceObject();
    ArrayView<point_t> v_points(points);
    auto mbr = mbr_;

    thrust::for_each(thrust::cuda::par.on(stream.cuda_stream()),
                     thrust::make_counting_iterator<uint32_t>(0),
                     thrust::make_counting_iterator<uint32_t>(points.size()),
                     [=] __device__(uint32_t point_id) mutable {
                       const auto& p = v_points[point_id];
                       if (mbr.Contains(p) && d_grid.IsOverlapping(p)) {
                         d_overlapping.Append(point_id);
                       } else {
                         d_no_overlapping.Append(point_id);
                       }
                     });
  }

  thrust::device_vector<mbr_t> GetTightCellMbrs(
      const Stream& stream, const thrust::device_vector<point_t>& points) {
    auto n_cells = cell_ids_.size();
    thrust::device_vector<mbr_t> mbrs(n_cells);
    auto* p_mbrs = thrust::raw_pointer_cast(mbrs.data());
    auto* p_cell_ids = thrust::raw_pointer_cast(cell_ids_.data());
    auto* p_prefix_sum = thrust::raw_pointer_cast(prefix_sum_.data());
    auto* p_point_ids = thrust::raw_pointer_cast(point_ids_.data());
    auto* p_points = thrust::raw_pointer_cast(points.data());

    // TODO: Use thrust
    LaunchKernel(stream, [=] __device__() mutable {
      for (auto i = blockIdx.x; i < n_cells; i += gridDim.x) {
        auto begin = p_prefix_sum[i];
        auto end = p_prefix_sum[i + 1];
        auto n_points_in_cell = end - begin;
        auto& mbr = p_mbrs[i];

        for (auto j = threadIdx.x; j < n_points_in_cell; j += blockDim.x) {
          auto offset = begin + j;
          auto point_idx = p_point_ids[offset];
          const auto& p = p_points[point_idx];
          mbr.ExpandAtomic(p);
        }
      }
    });
    stream.Sync();
#if 0
    auto d_grid = DeviceObject();
    thrust::for_each(thrust::cuda::par.on(stream.cuda_stream()),
                     thrust::make_counting_iterator<uint32_t>(0),
                     thrust::make_counting_iterator<uint32_t>(cell_ids_.size()),
                     [=] __device__(uint32_t i) mutable {
                       auto cell_id = p_cell_ids[i];
                       auto begin = p_prefix_sum[i];
                       auto end = p_prefix_sum[i + 1];
                       auto& mbr = p_mbrs[i];

                       for (int offset = begin; offset < end; offset++) {
                         auto point_idx = p_point_ids[offset];
                         const auto& p = p_points[point_idx];
                         mbr.ExpandAtomic(p);
                       }
                       assert(d_grid.GetCellBounds(cell_id).Contains(mbr));
                     });
#endif
    return mbrs;
  }

  void GetTightCellMbrs(const Stream& stream,
                        const thrust::device_vector<point_t>& points,
                        thrust::device_vector<mbr_t>& mbrs) {
    auto n_cells = cell_ids_.size();
    mbrs.resize(n_cells);
    auto* p_mbrs = thrust::raw_pointer_cast(mbrs.data());
    auto* p_cell_ids = thrust::raw_pointer_cast(cell_ids_.data());
    auto* p_prefix_sum = thrust::raw_pointer_cast(prefix_sum_.data());
    auto* p_point_ids = thrust::raw_pointer_cast(point_ids_.data());
    auto* p_points = thrust::raw_pointer_cast(points.data());

    // TODO: Use thrust
    LaunchKernel(stream, [=] __device__() mutable {
      for (auto i = blockIdx.x; i < n_cells; i += gridDim.x) {
        auto begin = p_prefix_sum[i];
        auto end = p_prefix_sum[i + 1];
        auto n_points_in_cell = end - begin;
        auto& mbr = p_mbrs[i];

        for (auto j = threadIdx.x; j < n_points_in_cell; j += blockDim.x) {
          auto offset = begin + j;
          auto point_idx = p_point_ids[offset];
          const auto& p = p_points[point_idx];
          mbr.ExpandAtomic(p);
        }
      }
    });
#if 0
    auto d_grid = DeviceObject();
    thrust::for_each(thrust::cuda::par.on(stream.cuda_stream()),
                     thrust::make_counting_iterator<uint32_t>(0),
                     thrust::make_counting_iterator<uint32_t>(cell_ids_.size()),
                     [=] __device__(uint32_t i) mutable {
                       auto cell_id = p_cell_ids[i];
                       auto begin = p_prefix_sum[i];
                       auto end = p_prefix_sum[i + 1];
                       auto& mbr = p_mbrs[i];

                       for (int offset = begin; offset < end; offset++) {
                         auto point_idx = p_point_ids[offset];
                         const auto& p = p_points[point_idx];
                         mbr.ExpandAtomic(p);
                       }
                       assert(d_grid.GetCellBounds(cell_id).Contains(mbr));
                     });
#endif
  }

  ArrayView<uint32_t> get_prefix_sum() { return prefix_sum_; }

  ArrayView<uint32_t> get_point_ids() { return point_ids_; }

  void PrintHistogram() {
    auto n_nonempty = cell_ids_.size();
    hdr_histogram* histogram;
    // Initialise the histogram
    hdr_init(1,                     // Minimum value
             (int64_t) n_nonempty,  // Maximum value
             3,                     // Number of significant figures
             &histogram);           // Pointer to initialise
    thrust::host_vector<uint32_t> prefix_sum = prefix_sum_;
    for (uint32_t i = 0; i < n_nonempty; i++) {
      auto n_points = prefix_sum[i + 1] - prefix_sum[i];
      hdr_record_value(histogram,  // Histogram to record to
                       n_points);  // Value to record
    }
    FILE* file = fopen("/tmp/grid", "w");  // Open file in write mode
    CHECK(file != nullptr) << "Error opening file " << "/tmp/grid";
    hdr_percentiles_print(histogram,
                          file,  // File to write to
                          5,     // Granularity of printed values
                          1.0,   // Multiplier for results
                          CSV);  // Format CLASSIC/CSV supported.
    hdr_close(histogram);
    fclose(file);
    VLOG(1) << "Non empty cells " << n_nonempty << " Total cells "
            << get_grid_size() << " Occupancy "
            << (double) n_nonempty / get_grid_size();
    VLOG(1) << "Avg points/non empty cell "
            << prefix_sum[prefix_sum.size() - 1] / n_nonempty
            << " Avg points/all cells "
            << prefix_sum[prefix_sum.size() - 1] / get_grid_size();
  }

  void ComputeHistogram() {
    auto n_nonempty = cell_ids_.size();
    hdr_histogram* histogram;
    // Initialise the histogram
    hdr_init(1,                     // Minimum value
             (int64_t) n_nonempty,  // Maximum value
             3,                     // Number of significant figures
             &histogram);           // Pointer to initialise

    thrust::host_vector<uint32_t> prefix_sum = prefix_sum_;
    for (uint32_t i = 0; i < n_nonempty; i++) {
      auto n_points = prefix_sum[i + 1] - prefix_sum[i];
      hdr_record_value(histogram,  // Histogram to record to
                       n_points);  // Value to record
    }

    stats_["Histogram"] = DumpHistogram(histogram);
    hdr_close(histogram);
  }

  COORD_T GetCellDigonalLength() const {
    COORD_T sum2 = 0;

    for (int dim = 0; dim < N_DIMS; dim++) {
      auto extent = mbr_.get_extent(dim);
      auto unit =
          extent / reinterpret_cast<const uint32_t*>(&grid_size_.x)[dim];
      sum2 += unit * unit;
    }
    return sqrt(sum2);
  }

  const nlohmann::json& GetStats() const { return stats_; }

  const thrust::device_vector<uint64_t>& get_cell_ids() const {
    return cell_ids_;
  }

 private:
  Bitset<uint64_t> occupied_cells_;
  thrust::device_vector<uint64_t> cell_ids_;
  thrust::device_vector<uint32_t> prefix_sum_;  // offsets point to point ids
  thrust::device_vector<uint32_t> point_ids_;   // point ids
  nlohmann::json stats_;
  mbr_t mbr_;
  cell_idx_t grid_size_;
};
}  // namespace hd
#endif  // HAUSDORFF_DISTANCE_INDEX_UNIFORM_GRID_H
