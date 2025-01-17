
#ifndef HAUSDORFF_DISTANCE_CPU_H
#define HAUSDORFF_DISTANCE_CPU_H
#include <thrust/sort.h>

#include <thread>
#include <thrust/system/detail/generic/sort.inl>
#include <vector>

#include "distance.h"
#include "glog/logging.h"
#include "mbr.h"
#include "utils/type_traits.h"

namespace hd {
namespace detail {

DEV_HOST_INLINE
std::uint32_t expand_bits(std::uint32_t n) {
  //v = (v * 0x00010001u) & 0xFF0000FFu;
  //v = (v * 0x00000101u) & 0x0F00F00Fu;
  //v = (v * 0x00000011u) & 0xC30C30C3u;
  //v = (v * 0x00000005u) & 0x49249249u;
  //return v;
  n &= 0x0000ffff;                 
  n = (n | (n << 8)) & 0x00FF00FF; 
  n = (n | (n << 4)) & 0x0F0F0F0F; 
  n = (n | (n << 2)) & 0x33333333;  
  n = (n | (n << 1)) & 0x55555555;  
  return n;
}

DEV_HOST_INLINE
uint32_t morton_code(float2 xy, float resolution = 1024.0f) {
  xy.x = ::fminf(::fmaxf(xy.x * resolution, 0.0f), resolution - 1.0f);
  xy.y = ::fminf(::fmaxf(xy.y * resolution, 0.0f), resolution - 1.0f);
  auto xx = expand_bits(static_cast<uint32_t>(xy.x));
  auto yy = expand_bits(static_cast<uint32_t>(xy.y));
  return xx * 2 + yy;
}

DEV_HOST_INLINE
uint32_t morton_code(double2 xy, float resolution = 1024.0f) {
  xy.x = ::fmin(::fmax(xy.x * resolution, 0.0f), resolution - 1.0f);
  xy.y = ::fmin(::fmax(xy.y * resolution, 0.0f), resolution - 1.0f);
  auto xx = expand_bits(static_cast<uint32_t>(xy.x));
  auto yy = expand_bits(static_cast<uint32_t>(xy.y));
  return xx * 2 + yy;
}

DEV_HOST_INLINE
uint32_t morton_code(float3 xyz, float resolution = 1024.0f) {
  xyz.x = ::fminf(::fmaxf(xyz.x * resolution, 0.0f), resolution - 1.0f);
  xyz.y = ::fminf(::fmaxf(xyz.y * resolution, 0.0f), resolution - 1.0f);
  xyz.z = ::fminf(::fmaxf(xyz.z * resolution, 0.0f), resolution - 1.0f);
  auto xx = expand_bits(static_cast<uint32_t>(xyz.x));
  auto yy = expand_bits(static_cast<uint32_t>(xyz.y));
  auto zz = expand_bits(static_cast<uint32_t>(xyz.z));
  return xx * 4 + yy * 2 + zz;
}

DEV_HOST_INLINE
uint32_t morton_code(double3 xyz, float resolution = 1024.0f) {
  xyz.x = ::fmin(::fmax(xyz.x * resolution, 0.0f), resolution - 1.0f);
  xyz.y = ::fmin(::fmax(xyz.y * resolution, 0.0f), resolution - 1.0f);
  xyz.z = ::fmin(::fmax(xyz.z * resolution, 0.0f), resolution - 1.0f);
  auto xx = expand_bits(static_cast<uint32_t>(xyz.x));
  auto yy = expand_bits(static_cast<uint32_t>(xyz.y));
  auto zz = expand_bits(static_cast<uint32_t>(xyz.z));
  return xx * 4 + yy * 2 + zz;
}

template <typename T>
void update_maximum(std::atomic<T>& maximum_value, T const& value) noexcept {
  T prev_value = maximum_value;
  while (prev_value < value &&
         !maximum_value.compare_exchange_weak(prev_value, value)) {}
}
}  // namespace detail

template <typename POINT_T>
typename vec_info<POINT_T>::type CalculateHausdorffDistance(
    std::vector<POINT_T>& points_a, std::vector<POINT_T>& points_b) {
  using coord_t = typename vec_info<POINT_T>::type;
  coord_t cmax = 0;
  std::random_device rd;  // Seed source
  std::mt19937 g(rd());   // Mersenne Twister engine seeded with rd()

  // Shuffle the vector
  std::shuffle(points_a.begin(), points_a.end(), g);
  std::shuffle(points_b.begin(), points_b.end(), g);
  uint32_t compared_pairs = 0;

  for (size_t i = 0; i < points_a.size(); i++) {
    coord_t cmin = std::numeric_limits<coord_t>::max();

    for (size_t j = 0; j < points_b.size(); j++) {
      auto d = EuclideanDistance2(points_a[i], points_b[j]);
      if (d < cmin) {
        cmin = d;
      }
      compared_pairs++;
      if (cmin <= cmax) {
        break;
      }
    }
    if (cmin != std::numeric_limits<coord_t>::max() && cmin > cmax) {
      cmax = cmin;
    }
  }
  LOG(INFO) << "Compared Pairs: " << compared_pairs;
  return sqrt(cmax);
}

template <typename POINT_T>
typename vec_info<POINT_T>::type CalculateHausdorffDistanceParallel(
    std::vector<POINT_T>& points_a, std::vector<POINT_T>& points_b) {
  using coord_t = typename vec_info<POINT_T>::type;
  std::random_device rd;  // Seed source
  std::mt19937 g(rd());   // Mersenne Twister engine seeded with rd()

  // Shuffle the vector
  std::shuffle(points_a.begin(), points_a.end(), g);
  std::shuffle(points_b.begin(), points_b.end(), g);

  std::vector<std::thread> threads;
  auto thread_count = std::thread::hardware_concurrency();
  auto avg_points = (points_a.size() + thread_count - 1) / thread_count;
  std::atomic<coord_t> cmax;
  std::atomic_uint64_t compared_pairs{0};

  cmax = 0;

  for (int tid = 0; tid < thread_count; tid++) {
    threads.emplace_back(std::thread([&, tid]() {
      auto begin = tid * avg_points;
      auto end = std::min(begin + avg_points, points_a.size());
      uint64_t local_compared_pairs = 0;

      for (int i = begin; i < end; i++) {
        auto cmin = std::numeric_limits<coord_t>::max();
        for (size_t j = 0; j < points_b.size(); j++) {
          auto d = EuclideanDistance2(points_a[i], points_b[j]);
          local_compared_pairs++;
          if (d < cmin) {
            cmin = d;
          }
          if (cmin < cmax) {
            break;
          }
        }
        if (cmin != std::numeric_limits<coord_t>::max()) {
          detail::update_maximum(cmax, cmin);
        }
      }
      compared_pairs += local_compared_pairs;
    }));
  }

  for (auto& thread : threads) {
    thread.join();
  }
  LOG(INFO) << "Compared Pairs: " << compared_pairs;
  return sqrt(cmax);
}

template <typename POINT_T>
typename vec_info<POINT_T>::type CalculateHausdorffDistanceZOrder(
    std::vector<POINT_T>& points_a, std::vector<POINT_T>& points_b) {
  using coord_t = typename vec_info<POINT_T>::type;
  constexpr int n_dims = vec_info<POINT_T>::n_dims;
  using mbr_t = Mbr<coord_t, n_dims>;
  coord_t cmax = 0;
  mbr_t mbr;

  for (auto& p : points_a) {
    mbr.Expand(p);
  }

  for (auto& p : points_b) {
    mbr.Expand(p);
  }

  auto comp = [mbr](const POINT_T& a, const POINT_T& b) {
    auto np_a = mbr.Normalize(a);
    auto np_b = mbr.Normalize(b);

    return detail::morton_code(np_a) < detail::morton_code(np_b);
  };
  std::sort(points_a.begin(), points_a.end(), comp);
  std::sort(points_b.begin(), points_b.end(), comp);

  uint32_t compared_pairs = 0;
  uint32_t dc = points_b.size() / 2;

  for (size_t i = 0; i < points_a.size(); i++) {
    coord_t cmin = std::numeric_limits<coord_t>::max();

    for (int j = dc, k = dc; j >= 0 || k < points_b.size();) {
      auto dist_l = std::numeric_limits<coord_t>::max();
      auto dist_r = std::numeric_limits<coord_t>::max();

      if (j >= 0) {
        dist_l = EuclideanDistance2(points_a[i], points_b[j]);
        --j;
        compared_pairs++;
      }
      if (k < points_b.size()) {
        dist_r = EuclideanDistance2(points_a[i], points_b[k]);
        ++k;
        compared_pairs++;
      }
      double tmp = std::min(cmin, std::min(dist_l, dist_r));
	  if(tmp!=cmin){
		cmin = tmp;
        dc = dist_l < dist_r ? j : k;
	  }
      if (cmin <= cmax) {
        dc = dist_l < dist_r ? j : k;
        break;
      }
    }

    if (cmin != std::numeric_limits<coord_t>::max()) {
      cmax = std::max(cmax, cmin);
    }
  }
  LOG(INFO) << "Compared Pairs: " << compared_pairs;
  return sqrt(cmax);
}

template <typename POINT_T>
typename vec_info<POINT_T>::type CalculateHausdorffDistanceYuan(
    std::vector<POINT_T>& points_a, std::vector<POINT_T>& points_b) {
  using coord_t = typename vec_info<POINT_T>::type;
  coord_t cmax = 0;

  double minx = std::numeric_limits<coord_t>::max();
  double miny = std::numeric_limits<coord_t>::max();
  double maxx = std::numeric_limits<coord_t>::min();
  double maxy = std::numeric_limits<coord_t>::min();
  for(int i=0; i<points_a.size(); i++){
    auto p = points_a[i];
	if(p.x < minx)minx = p.x;
	if(p.y < miny)miny = p.y;
	if(p.x > maxx)maxx = p.x;
	if(p.y > maxy)maxy = p.y;
  }

  for(int i=0; i<points_a.size(); i++){
    auto p = points_b[i];
	if(p.x < minx)minx = p.x;
	if(p.y < miny)miny = p.y;
	if(p.x > maxx)maxx = p.x;
	if(p.y > maxy)maxy = p.y;
  }
  
  auto comp = [minx, miny, maxx, maxy](const POINT_T& a, const POINT_T& b) {
    POINT_T np_a, np_b;
	np_a.x = (a.x-minx)/(maxx-minx);
	np_a.y = (a.y-miny)/(maxy-miny);
	np_b.x = (b.x-minx)/(maxx-minx);
	np_b.y = (b.y-miny)/(maxy-miny);

    return detail::morton_code(np_a) < detail::morton_code(np_b);
  };
  std::sort(points_a.begin(), points_a.end(), comp);
  std::sort(points_b.begin(), points_b.end(), comp);
  std::vector<uint32_t> Aloc(points_a.size());

  {
    int j = 0;
    for (int i = 0; i < points_b.size(); i++) {
      auto& b = points_b[i];
      auto& a = points_a[j];
      POINT_T p_a, p_b;
	  p_a.x = (a.x-minx)/(maxx-minx);
	  p_a.y = (a.y-miny)/(maxy-miny);
	  p_b.x = (b.x-minx)/(maxx-minx);
	  p_b.y = (b.y-miny)/(maxy-miny);

      if (detail::morton_code(p_b) >= detail::morton_code(p_a)) {
        Aloc[j] = i;
        j++;
        i--;
        if (j >= points_a.size())
          break;
      }
    }
    for (; j < points_a.size(); j++) {
      Aloc[j] = points_b.size() - 1;
    }
  }
  uint32_t compared_pairs = 0;

  for (int j = 0; j < points_a.size(); j++) {
    int loc = Aloc[j];
    auto cmin = std::numeric_limits<coord_t>::max();
    auto& tmp = points_a[j];

    for (int i = 0; i < points_b.size(); i++) {
      if (loc + i < points_b.size()) {
        auto d = EuclideanDistance2(tmp, points_b[loc + i]);
        if (d <= cmax) {
          cmin = 0;
          break;
        } else if (d < cmin) {
          cmin = d;
        }
        compared_pairs++;
      }
      if (loc >= i) {
        auto d = EuclideanDistance2(tmp, points_b[loc - i]);
        if (d <= cmax) {
          cmin = 0;
          break;
        } else if (d < cmin) {
          cmin = d;
        }
        compared_pairs++;
      }
      if (loc < i && loc + i >= points_b.size())
        break;
    }
    if (cmin > cmax)
      cmax = cmin;
  }

  //
  // for (size_t i = 0; i < points_a.size(); i++) {
  //   coord_t cmin = std::numeric_limits<coord_t>::max();
  //   auto dc = aloc[i];
  //
  //   for (int j = dc, k = dc; j >= 0 || k < points_b.size();) {
  //     auto dist_l = std::numeric_limits<coord_t>::max();
  //     auto dist_r = std::numeric_limits<coord_t>::max();
  //
  //     if (j >= 0) {
  //       dist_l = EuclideanDistance2(points_a[i], points_b[j]);
  //       --j;
  //       compared_pairs++;
  //     }
  //     if (k < points_b.size()) {
  //       dist_r = EuclideanDistance2(points_a[i], points_b[k]);
  //       ++k;
  //       compared_pairs++;
  //     }
  //     cmin = std::min(cmin, std::min(dist_l, dist_r));
  //     if (cmin <= cmax) {
  //       break;
  //     }
  //   }
  //
  //   if (cmin != std::numeric_limits<coord_t>::max()) {
  //     cmax = std::max(cmax, cmin);
  //   }
  // }
  LOG(INFO) << "Compared Pairs: " << compared_pairs;
  return sqrt(cmax);
}

}  // namespace hd
#endif  // HAUSDORFF_DISTANCE_CPU_H
