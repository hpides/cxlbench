#pragma once

#include "benchmark.hpp"
#include "parallel_benchmark.hpp"
#include "single_benchmark.hpp"

namespace cxlbench {

static constexpr auto CONFIG_FILE_EXTENSION = ".yaml";

class BenchmarkFactory {
 public:
  static std::vector<YAML::Node> get_config_files(const std::filesystem::path& config_file_path);

  static std::vector<SingleBenchmark> create_single_benchmarks(std::vector<YAML::Node>& configs);

  static std::vector<ParallelBenchmark> create_parallel_benchmarks(std::vector<YAML::Node>& configs);

 private:
  static std::vector<BenchmarkConfig> create_benchmark_matrix(YAML::Node& config_args, YAML::Node& matrix_args);

  static void parse_yaml_node(std::vector<BenchmarkConfig>& bm_configs, YAML::iterator& parallel_bm_it,
                              std::string& unique_name);
};

}  // namespace cxlbench
