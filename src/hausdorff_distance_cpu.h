
#ifndef HAUSDORFF_DISTANCE_CPU_H
#define HAUSDORFF_DISTANCE_CPU_H
#include <thrust/sort.h>

#include <thread>
#include <thrust/system/detail/generic/sort.inl>
#include <vector>

#include "distance.h"
#include "glog/logging.h"
#include "mbr.h"
#include "morton_code.h"
#include "utils/type_traits.h"

namespace hd {
namespace detail {

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
      auto prev_cmax = cmax;
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

  cmax = 0;

  for (int tid = 0; tid < thread_count; tid++) {
    threads.emplace_back(std::thread([&, tid]() {
      auto begin = tid * avg_points;
      auto end = std::min(begin + avg_points, points_a.size());

      for (int i = begin; i < end; i++) {
        auto cmin = std::numeric_limits<coord_t>::max();
        for (size_t j = 0; j < points_b.size(); j++) {
          auto d = EuclideanDistance2(points_a[i], points_b[j]);
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
    }));
  }

  for (auto& thread : threads) {
    thread.join();
  }
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
      if (tmp != cmin) {
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
  std::vector<uint32_t> Aloc(points_a.size());

  {
    int j = 0;
    for (int i = 0; i < points_b.size(); i++) {
      auto& b = points_b[i];
      auto& a = points_a[j];
      auto np_a = mbr.Normalize(a);
      auto np_b = mbr.Normalize(b);

      if (detail::morton_code(np_b) >= detail::morton_code(np_a)) {
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
