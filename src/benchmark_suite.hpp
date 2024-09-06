#pragma once

#include <chrono>

#include "benchmark.hpp"
#include "types.hpp"
#include "utils.hpp"

namespace cxlbench {

struct BenchOptions {
  const std::filesystem::path& config_file;
  const std::filesystem::path& result_directory;
  // For parallel workloads, ensure that only these configurations are executed in which the thread count matches.
  const bool only_equal_thread_counts;
  const TimePointMS start_timestamp;
  // Delay after starting a benchmark case before another one starts. The second benchmark case does not start before
  // start_timestamp + delay.
  const std::chrono::milliseconds delay;
};

class BenchmarkSuite {
 public:
  static void run_benchmarks(const BenchOptions& options);
};

}  // namespace cxlbench
