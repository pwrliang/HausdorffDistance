#ifndef GRID_H
#define GRID_H

#include <optix.h>
#include <thrust/binary_search.h>
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sort.h>
#include <thrust/transform_reduce.h>

#include <iostream>
#include <nlohmann/json.hpp>

#include "glog/logging.h"
#include "hdr/hdr_histogram.h"
#include "mbr.h"
#include "utils/array_view.h"
#include "utils/bitset.h"
#include "utils/launcher.h"
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

// Weighted index term: (2i - n + 1) * x_i
struct weighted_index_term_uint {
  const uint32_t* data;
  uint32_t n;

  weighted_index_term_uint(const uint32_t* _data, uint32_t _n)
      : data(_data), n(_n) {}

  __host__ __device__ float operator()(uint32_t i) const {
    return (2.0f * i - n + 1) * data[i];
  }
};

}  // namespace details

template <typename COORD_T, int N_DIMS>
class Grid {
  using point_t = typename cuda_vec<COORD_T, N_DIMS>::type;
  using mbr_t = Mbr<COORD_T, N_DIMS>;
  using cell_pos_t = typename cuda_vec<uint32_t, N_DIMS>::type;

 public:
  Grid() = default;

  DEV_HOST Grid(const mbr_t& mbr, const cell_pos_t& grid_size,
                const dev::Bitset<uint64_t>& occupied_cells)
      : mbr_(mbr), grid_size_(grid_size), occupied_cells_(occupied_cells) {}

  DEV_INLINE bool Insert(const point_t& p) {
    auto cell_p = CalculateCellPos(p);
    auto cell_idx = EncodeCellPos(cell_p);
    auto cell_mbr = GetCellBounds(cell_idx);
    assert(cell_mbr.Contains(p));
    return occupied_cells_.set_bit_atomic(cell_idx);
  }

  DEV_INLINE mbr_t GetCellBounds(const cell_pos_t& cell_pos) const {
    point_t lower_p, upper_p;

    for (int dim = 0; dim < N_DIMS; dim++) {
      auto extent = mbr_.get_extent(dim);
      auto grid_size = get_grid_size(dim);
      auto unit = extent / grid_size;
      auto idx = reinterpret_cast<const unsigned int*>(&cell_pos.x)[dim];
      assert(idx < grid_size);
      auto begin = unit * idx + mbr_.lower(dim);
      auto end = std::min(begin + unit, mbr_.upper(dim));
      begin = nextafter(begin, begin - 1);
      begin = nextafter(begin, begin - 1);
      end = nextafter(end, end + 1);
      end = nextafter(end, end + 1);
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

  DEV_HOST_INLINE uint64_t EncodeCellPos(const cell_pos_t& pos) const {
    return details::EncodeCellPos<N_DIMS>(pos, grid_size_);
  }

  DEV_HOST_INLINE cell_pos_t DecodeCellIdx(uint64_t cell_idx) const {
    return details::DecodeCellIdx<N_DIMS>(cell_idx, grid_size_);
  }

 private:
  mbr_t mbr_;
  cell_pos_t grid_size_;
  dev::Bitset<uint64_t> occupied_cells_;
};

}  // namespace dev

template <typename COORD_T, int N_DIMS>
class Grid {
  using point_t = typename cuda_vec<COORD_T, N_DIMS>::type;
  using mbr_t = Mbr<COORD_T, N_DIMS>;
  using cell_pos_t = typename cuda_vec<uint32_t, N_DIMS>::type;

  static_assert(N_DIMS == 2 || N_DIMS == 3, "Invalid N_DIMS");

 public:
  Grid() = default;

  typename cuda_vec<unsigned int, N_DIMS>::type CalculateGridResolution(
      const mbr_t& mbr, unsigned int n_points, int n_points_per_cell) {
    double volume = 1;
    for (int dim = 0; dim < N_DIMS; dim++) {
      volume *= mbr.get_extent(dim);
    }
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

  void Init(const cell_pos_t& grid_size, const mbr_t& mbr) {
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

  void Insert(const Stream& stream,
              const thrust::device_vector<point_t>& points) {
    auto d_grid = DeviceObject();
    thrust::device_vector<uint32_t> n_primitives;
    SharedValue<uint32_t> tmp_offset;

    occupied_cells_.Clear(stream.cuda_stream());
    tmp_offset.set(stream.cuda_stream(), 0);

    thrust::for_each(thrust::cuda::par.on(stream.cuda_stream()), points.begin(),
                     points.end(), [=] __device__(const point_t& p) mutable {
                       d_grid.Insert(p);
                     });

    auto non_empty_cells =
        occupied_cells_.GetPositiveCount(stream.cuda_stream());

    stats_["TotalCells"] = get_grid_size();
    stats_["NonEmptyCells"] = non_empty_cells;

    cell_ids_.resize(non_empty_cells);
    n_primitives.resize(non_empty_cells, 0);
    prefix_sum_.resize(non_empty_cells + 1, 0);

    auto* p_cell_ids = thrust::raw_pointer_cast(cell_ids_.data());
    auto* p_n_primitives = thrust::raw_pointer_cast(n_primitives.data());
    auto* p_offset = tmp_offset.data();
    auto d_occupied_cells = occupied_cells_.DeviceObject();

    // collect non-empty cell ids
    thrust::for_each(
        thrust::cuda::par.on(stream.cuda_stream()),
        thrust::make_counting_iterator<size_t>(0),
        thrust::make_counting_iterator<size_t>(occupied_cells_.GetSize()),
        [=] __device__(size_t cell_idx) mutable {
          if (d_occupied_cells.get_bit(cell_idx)) {
            p_cell_ids[atomicAdd(p_offset, 1)] = cell_idx;
          }
        });

    // sort cell ids for binary search
    thrust::sort(thrust::cuda::par.on(stream.cuda_stream()), cell_ids_.begin(),
                 cell_ids_.end());
    // count the number of primitives per cell
    thrust::for_each(thrust::cuda::par.on(stream.cuda_stream()), points.begin(),
                     points.end(), [=] __device__(const point_t& p) mutable {
                       auto cell_pos = d_grid.CalculateCellPos(p);
                       auto cell_idx = d_grid.EncodeCellPos(cell_pos);
                       auto it = thrust::lower_bound(
                           thrust::seq, p_cell_ids,
                           p_cell_ids + non_empty_cells, cell_idx);
                       auto cell_renumbered_idx = it - p_cell_ids;
                       assert(cell_renumbered_idx >= 0 &&
                              cell_renumbered_idx < non_empty_cells);
                       atomicAdd(&p_n_primitives[cell_renumbered_idx], 1);
                     });

    SharedValue<uint32_t> min_points, max_points;
    auto* p_min_points = min_points.data();
    auto* p_max_points = max_points.data();

    min_points.set(stream.cuda_stream(), std::numeric_limits<uint32_t>::max());
    max_points.set(stream.cuda_stream(), 0);

    thrust::for_each(thrust::cuda::par.on(stream.cuda_stream()),
                     n_primitives.begin(), n_primitives.end(),
                     [=] __device__(uint32_t n) {
                       assert(n > 0);
                       atomicMin(p_min_points, n);
                       atomicMax(p_max_points, n);
                     });
    stats_["MinPoints"] = min_points.get(stream.cuda_stream());
    stats_["MaxPoints"] = max_points.get(stream.cuda_stream());
    stats_["AvgPoints"] = points.size() / non_empty_cells;
    stats_["GiniIndex"] = gini_index_thrust(stream, n_primitives);

    thrust::inclusive_scan(thrust::cuda::par.on(stream.cuda_stream()),
                           n_primitives.begin(), n_primitives.end(),
                           prefix_sum_.begin() + 1);
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
        [=] __device__(const thrust::tuple<uint32_t, point_t>& tuple) mutable {
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
    thrust::for_each(thrust::cuda::par.on(stream.cuda_stream()),
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

  dev::Grid<COORD_T, N_DIMS> DeviceObject() {
    return dev::Grid<COORD_T, N_DIMS>(mbr_, grid_size_,
                                      occupied_cells_.DeviceObject());
  }

  static uint32_t EstimatedMaxCells(uint32_t memory_quota_mb) {
    return (uint64_t) memory_quota_mb * 1024 * 1024 / sizeof(uint32_t);
  }

  thrust::device_vector<mbr_t> GetCellMbrs(const Stream& stream) {
    thrust::device_vector<mbr_t> mbrs(cell_ids_.size());
    auto d_grid = DeviceObject();

    thrust::transform(thrust::cuda::par.on(stream.cuda_stream()),
                      cell_ids_.begin(), cell_ids_.end(), mbrs.begin(),
                      [=] __device__(uint64_t cell_id) {
                        return d_grid.GetCellBounds(cell_id);
                      });
    return mbrs;
  }

  thrust::device_vector<mbr_t> GetTightCellMbrs(
      const Stream& stream, const thrust::device_vector<point_t>& points) {
    thrust::device_vector<mbr_t> mbrs(cell_ids_.size());
    auto* p_mbrs = thrust::raw_pointer_cast(mbrs.data());
    auto* p_cell_ids = thrust::raw_pointer_cast(cell_ids_.data());
    auto* p_prefix_sum = thrust::raw_pointer_cast(prefix_sum_.data());
    auto* p_point_ids = thrust::raw_pointer_cast(point_ids_.data());
    auto* p_points = thrust::raw_pointer_cast(points.data());
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

    return mbrs;
  }

  const thrust::device_vector<uint32_t>& get_point_ids() const {
    return point_ids_;
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
                          3,     // Granularity of printed values
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

  nlohmann::json GetStats() const { return stats_; }

 private:
  Bitset<uint64_t> occupied_cells_;
  thrust::device_vector<uint64_t> cell_ids_;
  thrust::device_vector<uint32_t> prefix_sum_;  // offsets point to point ids
  thrust::device_vector<uint32_t> point_ids_;   // point ids
  nlohmann::json stats_;
  mbr_t mbr_;
  cell_pos_t grid_size_;

  // Gini index function for unsigned int input
  float gini_index_thrust(const Stream& stream,
                          thrust::device_vector<uint32_t> d_values) {
    uint32_t n = d_values.size();
    if (n == 0)
      return 0.0f;

    // Sort values
    thrust::sort(thrust::cuda::par.on(stream.cuda_stream()), d_values.begin(),
                 d_values.end());

    // Total sum of values
    auto total = thrust::transform_reduce(
        thrust::cuda::par.on(stream.cuda_stream()), d_values.begin(),
        d_values.end(), thrust::identity<uint32_t>(), 0.0f,
        thrust::plus<float>());

    if (total == 0)
      return 0.0f;

    // Compute weighted sum
    auto* raw_ptr = thrust::raw_pointer_cast(d_values.data());
    auto weighted_sum = thrust::transform_reduce(
        thrust::cuda::par.on(stream.cuda_stream()),
        thrust::counting_iterator<uint32_t>(0),
        thrust::counting_iterator<uint32_t>(n),
        dev::details::weighted_index_term_uint(raw_ptr, n), 0.0f,
        thrust::plus<float>());

    return weighted_sum / (n * total);
  }
};
}  // namespace hd
#endif  // GRID_H
