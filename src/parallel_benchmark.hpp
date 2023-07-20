#pragma once

#include <string>

#include "benchmark.hpp"

namespace mema {

class ParallelBenchmark : public Benchmark {
 public:
  /**
   * Constructor for two writing benchmarks, i.e., no reusage of existing files.
   */
  ParallelBenchmark(const std::string& benchmark_name, std::string first_benchmark_name,
                    std::string second_benchmark_name, const BenchmarkConfig& first_config,
                    const BenchmarkConfig& second_config, std::vector<std::unique_ptr<BenchmarkExecution>>&& executions,
                    std::vector<std::unique_ptr<BenchmarkResult>>&& results);

  /**
   * Constructor for one writing benchmark and one read-only benchmark.
   * Only reuse the read-only file.
   */
  ParallelBenchmark(const std::string& benchmark_name, std::string first_benchmark_name,
                    std::string second_benchmark_name, const BenchmarkConfig& first_config,
                    const BenchmarkConfig& second_config, std::vector<std::unique_ptr<BenchmarkExecution>>&& executions,
                    std::vector<std::unique_ptr<BenchmarkResult>>&& results, std::filesystem::path pmem_file_first);

  /**
   * Constructor for two read-only benchmarks.
   * Reuse both files.
   */
  ParallelBenchmark(const std::string& benchmark_name, std::string first_benchmark_name,
                    std::string second_benchmark_name, const BenchmarkConfig& first_config,
                    const BenchmarkConfig& second_config, std::vector<std::unique_ptr<BenchmarkExecution>>&& executions,
                    std::vector<std::unique_ptr<BenchmarkResult>>&& results, std::filesystem::path pmem_file_first,
                    std::filesystem::path pmem_file_second);

  ParallelBenchmark(ParallelBenchmark&& other) = default;
  ParallelBenchmark(const ParallelBenchmark& other) = delete;
  ParallelBenchmark& operator=(const ParallelBenchmark& other) = delete;
  ParallelBenchmark& operator=(ParallelBenchmark&& other) = delete;

  bool run() final;

  void generate_data() final;

  void set_up() final;

  nlohmann::json get_result_as_json() final;

  const std::string& get_benchmark_name_one() const;
  const std::string& get_benchmark_name_two() const;

  ~ParallelBenchmark() { ParallelBenchmark::tear_down(false); }

 private:
  const std::string benchmark_name_one_;
  const std::string benchmark_name_two_;
};

}  // namespace mema
