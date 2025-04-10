#ifndef RUNNING_STATS_H
#define RUNNING_STATS_H
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>

#include "glog/logging.h"
#include "hdr/hdr_histogram.h"

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

#endif  // RUNNING_STATS_H
