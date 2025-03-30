#ifndef RUNNING_STATS_H
#define RUNNING_STATS_H
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>

#include "glog/logging.h"

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

#endif  // RUNNING_STATS_H
