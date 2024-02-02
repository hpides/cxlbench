#pragma once

#include "benchmark.hpp"
#include "utils.hpp"

namespace mema {

struct MemaOptions {
  const std::filesystem::path& config_file;
  const std::filesystem::path& result_directory;
  // For parallel workloads, ensure that only these configurations are executed in which the thread count matches.
  const bool only_equal_thread_counts;
};

class BenchmarkSuite {
 public:
  static void run_benchmarks(const MemaOptions& options);
};

}  // namespace mema
