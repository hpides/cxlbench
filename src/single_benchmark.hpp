#pragma once

#include <string>

#include "benchmark.hpp"

namespace mema {

class SingleBenchmark : public Benchmark {
 public:
  SingleBenchmark(const std::string& benchmark_name, const BenchmarkConfig& config,
                  std::vector<std::unique_ptr<BenchmarkExecution>>&& executions,
                  std::vector<std::unique_ptr<BenchmarkResult>>&& results);
  SingleBenchmark(const std::string& benchmark_name, const BenchmarkConfig& config,
                  std::vector<std::unique_ptr<BenchmarkExecution>>&& executions,
                  std::vector<std::unique_ptr<BenchmarkResult>>&& results, std::filesystem::path pmem_file);

  SingleBenchmark(SingleBenchmark&& other) = default;
  SingleBenchmark(const SingleBenchmark& other) = delete;
  SingleBenchmark& operator=(const SingleBenchmark& other) = delete;
  SingleBenchmark& operator=(SingleBenchmark&& other) = delete;

  bool run() final;

  void generate_data() final;

  void set_up() final;

  nlohmann::json get_result_as_json() final;

  ~SingleBenchmark() { SingleBenchmark::tear_down(false); }
};

}  // namespace mema
