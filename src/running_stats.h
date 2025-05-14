#ifndef RUNNING_STATS_H
#define RUNNING_STATS_H
#include <glog/logging.h>
#include <hdr/hdr_histogram.h>
#include <thrust/sort.h>
#include <thrust/transform_reduce.h>

#include <fstream>
#include <nlohmann/json.hpp>
#include <string>

#include "utils/stream.h"
namespace hd {

namespace dev {
namespace details {

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
}  // namespace dev
// Gini index function for unsigned int input
inline float gini_index_thrust(const Stream& stream,
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

class RunningStats {
 public:
  // Access the singleton instance
  static RunningStats& instance() {
    static RunningStats inst;
    return inst;
  }

  // Log a key-value pair
  template <typename T>
  void Log(const std::string& key, const T& value) {
    stats_[key] = value;
  }
  nlohmann::json& Log(const std::string& key) { return stats_[key]; }

  // Retrieve the JSON object
  const nlohmann::json& Get() const { return stats_; }

  const nlohmann::json& Get(const std::string& key) const {
    return stats_.at(key);
  }

  // Dump JSON to string (with optional file output)
  void Dump(const std::string& path, int indent = 4) const {
    std::string result = stats_.dump(indent);
    if (!path.empty()) {
      std::ofstream out(path);
      CHECK(out.is_open()) << "Cannot open " << path;
      out << result;
      out.close();
    }
  }

 private:
  RunningStats() = default;
  ~RunningStats() = default;
  RunningStats(const RunningStats&) = delete;
  RunningStats& operator=(const RunningStats&) = delete;

  nlohmann::json stats_;
};

inline nlohmann::json DumpHistogram(hdr_histogram* histogram,
                                    int ticks_per_half_distance = 3) {
  auto j = nlohmann::json::array();
  char* buffer = nullptr;
  size_t size = 0;
  FILE* memstream = open_memstream(&buffer, &size);
  if (!memstream) {
    perror("open_memstream failed");
    return j;
  }

  hdr_percentiles_print(histogram, memstream, ticks_per_half_distance, 1.0,
                        CLASSIC);
  fclose(memstream);

  // Parse to JSON
  std::istringstream iss(buffer);
  std::string line;

  while (std::getline(iss, line)) {
    if (line.empty() || line[0] == '-' ||
        line.find("Value") != std::string::npos)
      continue;

    std::istringstream ls(line);
    double value, percentile;
    int count;
    if (ls >> value >> percentile >> count) {
      j.push_back(
          {{"value", value}, {"percentile", percentile}, {"count", count}});
    }
  }

  free(buffer);
  return j;
}
}  // namespace hd

#endif  // RUNNING_STATS_H
