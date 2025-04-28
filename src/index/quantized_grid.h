#ifndef HAUSDORFF_DISTANCE_INDEX_QUANTIZED_GRID_H
#define HAUSDORFF_DISTANCE_INDEX_QUANTIZED_GRID_H
#include <thrust/device_vector.h>

#include <cassert>
#include <cstdint>

#include "utils/bitset.h"
#include "utils/stream.h"
#include "utils/type_traits.h"
#include "utils/util.h"

namespace hd {
namespace dev {
namespace details {

template <int N_DIMS>
DEV_HOST_INLINE uint64_t
EncodeCellPos(const typename cuda_vec<int, N_DIMS>::type& cell_pos,
              const typename cuda_vec<int, N_DIMS>::type& grid_size) {
  assert(false);
  return -1;
}

template <>
DEV_HOST_INLINE uint64_t
EncodeCellPos<2>(const typename cuda_vec<int, 2>::type& cell_pos,
                 const typename cuda_vec<int, 2>::type& grid_size) {
  assert(cell_pos.x >= 0 && cell_pos.y >= 0);
  assert(cell_pos.x < grid_size.x && cell_pos.y < grid_size.y);
  return (uint64_t) cell_pos.y * grid_size.x + cell_pos.x;
}

template <>
DEV_HOST_INLINE uint64_t
EncodeCellPos<3>(const typename cuda_vec<int, 3>::type& cell_pos,
                 const typename cuda_vec<int, 3>::type& grid_size) {
  assert(cell_pos.x >= 0 && cell_pos.y >= 0 && cell_pos.z >= 0);
  assert(cell_pos.x < grid_size.x && cell_pos.y < grid_size.y &&
         cell_pos.z < grid_size.z);
  return (uint64_t) cell_pos.z * grid_size.x * grid_size.y +
         (uint64_t) cell_pos.y * grid_size.x + cell_pos.x;
}

template <int N_DIMS>
DEV_HOST_INLINE typename cuda_vec<int, N_DIMS>::type DecodeCellIdx(
    uint64_t cell_idx, const typename cuda_vec<int, N_DIMS>::type& grid_size) {
  assert(false);
  return typename cuda_vec<int, N_DIMS>::type();
}

template <>
DEV_HOST_INLINE int2 DecodeCellIdx<2>(uint64_t cell_idx,
                                      const int2& grid_size) {
  int2 cell_pos;
  cell_pos.x = cell_idx % grid_size.x;
  cell_pos.y = cell_idx / grid_size.x;
  return cell_pos;
}

template <>
DEV_HOST_INLINE int3 DecodeCellIdx<3>(uint64_t cell_idx,
                                      const int3& grid_size) {
  int3 cell_pos;
  cell_pos.z = cell_idx / ((uint64_t) grid_size.x * grid_size.y);
  cell_idx %= (uint64_t) grid_size.x * grid_size.y;
  cell_pos.x = cell_idx % grid_size.x;
  cell_pos.y = cell_idx / grid_size.x;
  return cell_pos;
}
}  // namespace details

template <typename COORD_T, int N_DIMS>
class QuantizedGrid {
  using point_t = typename cuda_vec<COORD_T, N_DIMS>::type;
  using mbr_t = Mbr<COORD_T, N_DIMS>;
  using quantized_point_t = typename cuda_vec<int, N_DIMS>::type;

 public:
  QuantizedGrid() = default;

  DEV_HOST QuantizedGrid(const mbr_t& mbr, COORD_T cell_size,
                         const quantized_point_t& resolution,
                         const dev::Bitset<uint64_t>& occupied_cells)
      : mbr_(mbr),
        cell_size_(cell_size),
        resolution_(resolution),
        occupied_cells_(occupied_cells) {}

  DEV_HOST_INLINE quantized_point_t QuantizePoint(const point_t& p) const {
    quantized_point_t cell_idx;

    for (int dim = 0; dim < N_DIMS; dim++) {
      auto val = reinterpret_cast<const COORD_T*>(&p.x)[dim];
      auto min = mbr_.lower(dim);
      auto diff = val - min;
      int pos = diff / cell_size_;
      auto lim = reinterpret_cast<const int*>(&resolution_.x)[dim];

      reinterpret_cast<int*>(&cell_idx.x)[dim] = pos;
    }
    return cell_idx;
  }

  // DEV_HOST_INLINE point_t CalculatePoint(const quantized_point_t& cell_idx)
  // const {
  //   point_t p;
  //
  //   for (int dim = 0; dim < N_DIMS; dim++) {
  //     reinterpret_cast<COORD_T*>(&p.x)[dim] =
  //         reinterpret_cast<const int*>(&cell_idx.x)[dim] * cell_size_;
  //   }
  //   return p;
  // }

  DEV_HOST_INLINE uint64_t EncodeCellPos(const quantized_point_t& pos) const {
    return details::EncodeCellPos<N_DIMS>(pos, resolution_);
  }

  DEV_HOST_INLINE quantized_point_t DecodeCellIdx(uint64_t cell_idx) const {
    return details::DecodeCellIdx<N_DIMS>(cell_idx, resolution_);
  }

  DEV_INLINE bool Insert(const point_t& p) {
    auto cell_p = QuantizePoint(p);
    auto cell_idx = EncodeCellPos(cell_p);
    return occupied_cells_.set_bit_atomic(cell_idx);
  }

 private:
  mbr_t mbr_;
  COORD_T cell_size_;
  quantized_point_t resolution_;
  dev::Bitset<uint64_t> occupied_cells_;
};

}  // namespace dev

template <typename COORD_T, int N_DIMS>
class QuantizedGrid {
  using point_t = typename cuda_vec<COORD_T, N_DIMS>::type;
  using mbr_t = Mbr<COORD_T, N_DIMS>;

  using device_object_t = dev::QuantizedGrid<COORD_T, N_DIMS>;

 public:
  using quantized_point_t = typename cuda_vec<int, N_DIMS>::type;

  QuantizedGrid() = default;

  QuantizedGrid(const mbr_t& mbr, int n_bits) : mbr_(mbr) {
    COORD_T l = 0;
    for (int dim = 0; dim < N_DIMS; dim++) {
      l = std::max(l, mbr.get_extent(dim));
    }
    auto n_cells = (1 << n_bits);
    cell_size_ = nextafter(l / n_cells, l);
    auto total_cells = 1;

    for (int dim = 0; dim < N_DIMS; dim++) {
      auto n = ceil(mbr.get_extent(dim) / cell_size_);
      reinterpret_cast<int*>(&resolution_.x)[dim] = n;

      total_cells *= n;
    }
    occupied_cells_.Init(total_cells);
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
                       auto cell_pos = d_grid.QuantizePoint(p);
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
          auto cell_pos = d_grid.QuantizePoint(p);
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
  }

  thrust::device_vector<quantized_point_t> GetRepresentativePoints(
      const Stream& stream) {
    thrust::device_vector<quantized_point_t> points(cell_ids_.size());
    auto d_grid = DeviceObject();

    thrust::transform(thrust::cuda::par.on(stream.cuda_stream()),
                      cell_ids_.begin(), cell_ids_.end(), points.begin(),
                      [=] __device__(uint64_t cell_idx) {
                        return d_grid.DecodeCellIdx(cell_idx);
                      });
    return points;
  }

  device_object_t DeviceObject() {
    return dev::QuantizedGrid<COORD_T, N_DIMS>(mbr_, cell_size_, resolution_,
                                               occupied_cells_.DeviceObject());
  }

  COORD_T GetDiagonalQuantizedLength() const { return sqrt(N_DIMS); }

  COORD_T GetDiagonalLength() const { return sqrt(N_DIMS) * cell_size_; }

  ArrayView<uint32_t> get_prefix_sum() { return prefix_sum_; }

  ArrayView<uint32_t> get_point_ids() { return point_ids_; }

 private:
  mbr_t mbr_;
  COORD_T cell_size_;
  quantized_point_t resolution_;
  Bitset<uint64_t> occupied_cells_;
  thrust::device_vector<uint64_t> cell_ids_;
  thrust::device_vector<uint32_t> prefix_sum_;  // offsets point to point ids
  thrust::device_vector<uint32_t> point_ids_;   // point ids
};
}  // namespace hd

#endif  // HAUSDORFF_DISTANCE_INDEX_QUANTIZED_GRID_H
