#pragma once

#include "benchmark.hpp"
#include "utils.hpp"

namespace mema {

struct MemaOptions {
  const std::filesystem::path& config_file;
  const std::filesystem::path& result_directory;
};

class BenchmarkSuite {
 public:
  static void run_benchmarks(const MemaOptions& options);
};

}  // namespace mema
