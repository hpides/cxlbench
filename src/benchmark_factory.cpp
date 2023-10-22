#include "benchmark_factory.hpp"

#include <yaml-cpp/yaml.h>

#include <string>

namespace mema {

std::vector<SingleBenchmark> BenchmarkFactory::create_single_benchmarks(std::vector<YAML::Node>& configs) {
  auto benchmarks = std::vector<SingleBenchmark>{};

  for (YAML::Node& config : configs) {
    for (YAML::iterator it = config.begin(); it != config.end(); ++it) {
      const auto name = it->first.as<std::string>();
      YAML::Node raw_bm_args = it->second;

      // Ignore parallel benchmarks
      YAML::Node parallel_bm = raw_bm_args["parallel_benchmark"];
      if (parallel_bm.IsDefined()) {
        continue;
      }

      YAML::Node bm_args = raw_bm_args["args"];
      if (!bm_args && raw_bm_args) {
        spdlog::critical("Benchmark config must contain 'args' if it is not empty.");
        utils::crash_exit();
      }

      YAML::Node bm_matrix = raw_bm_args["matrix"];
      if (bm_matrix) {
        std::vector<BenchmarkConfig> matrix = create_benchmark_matrix(bm_args, bm_matrix);
        for (BenchmarkConfig& bm : matrix) {
          std::vector<std::unique_ptr<BenchmarkExecution>> executions{};
          executions.push_back(std::make_unique<BenchmarkExecution>());

          std::vector<std::unique_ptr<BenchmarkResult>> results{};
          results.push_back(std::make_unique<BenchmarkResult>(bm));
          benchmarks.emplace_back(name, bm, std::move(executions), std::move(results));
        }
      } else {
        BenchmarkConfig bm_config = BenchmarkConfig::decode(bm_args);

        std::vector<std::unique_ptr<BenchmarkExecution>> executions{};
        executions.push_back(std::make_unique<BenchmarkExecution>());
        std::vector<std::unique_ptr<BenchmarkResult>> results{};
        results.push_back(std::make_unique<BenchmarkResult>(bm_config));

        benchmarks.emplace_back(name, bm_config, std::move(executions), std::move(results));
      }
    }
  }
  return benchmarks;
}

std::vector<ParallelBenchmark> BenchmarkFactory::create_parallel_benchmarks(std::vector<YAML::Node>& configs) {
  auto benchmarks = std::vector<ParallelBenchmark>{};

  for (YAML::Node& config : configs) {
    for (YAML::iterator it = config.begin(); it != config.end(); ++it) {
      const auto name = it->first.as<std::string>();
      YAML::Node raw_par_bm = it->second;

      // Only consider parallel nodes
      YAML::Node parallel_bm = raw_par_bm["parallel_benchmark"];

      if (!parallel_bm.IsDefined()) {
        continue;
      }

      if (parallel_bm.size() != 2) {
        spdlog::critical("Number of parallel benchmarks should be two.");
        utils::crash_exit();
      }

      std::vector<BenchmarkConfig> bm_one_configs{};
      std::vector<BenchmarkConfig> bm_two_configs{};
      std::string unique_name_one;
      std::string unique_name_two;

      YAML::iterator parallel_bm_it = parallel_bm.begin();
      parse_yaml_node(bm_one_configs, parallel_bm_it, unique_name_one);
      parallel_bm_it++;
      parse_yaml_node(bm_two_configs, parallel_bm_it, unique_name_two);

      // Build cartesian product of both benchmarks
      for (const BenchmarkConfig& config_one : bm_one_configs) {
        for (const BenchmarkConfig& config_two : bm_two_configs) {
          std::vector<std::unique_ptr<BenchmarkExecution>> executions{};
          executions.push_back(std::make_unique<BenchmarkExecution>());
          executions.push_back(std::make_unique<BenchmarkExecution>());

          std::vector<std::unique_ptr<BenchmarkResult>> results{};
          results.push_back(std::make_unique<BenchmarkResult>(config_one));
          results.push_back(std::make_unique<BenchmarkResult>(config_two));

          if (config_one.contains_write_op() && config_two.contains_write_op()) {
            benchmarks.emplace_back(name, unique_name_one, unique_name_two, config_one, config_two,
                                    std::move(executions), std::move(results));
          } else if (config_one.contains_write_op()) {
            // Reorder benchmarks if the first benchmark is read-only and the second writing
            std::swap(results[0], results[1]);
            benchmarks.emplace_back(name, unique_name_two, unique_name_one, config_two, config_one,
                                    std::move(executions), std::move(results));
          } else {
            benchmarks.emplace_back(name, unique_name_one, unique_name_two, config_one, config_two,
                                    std::move(executions), std::move(results));
          }
        }
      }
    }
  }
  return benchmarks;
}

void BenchmarkFactory::parse_yaml_node(std::vector<BenchmarkConfig>& bm_configs, YAML::iterator& parallel_bm_it,
                                       std::string& unique_name) {
  unique_name = parallel_bm_it->first.as<std::string>();
  YAML::Node raw_bm_args = parallel_bm_it->second;
  YAML::Node bm_args = raw_bm_args["args"];

  if (!bm_args) {
    spdlog::critical("Benchmark config must contain 'args' if it is not empty.");
    utils::crash_exit();
  }
  YAML::Node bm_matrix = raw_bm_args["matrix"];
  if (bm_matrix) {
    std::vector<BenchmarkConfig> matrix = create_benchmark_matrix(bm_args, bm_matrix);
    std::move(matrix.begin(), matrix.end(), std::back_inserter(bm_configs));
  } else {
    BenchmarkConfig bm_config = BenchmarkConfig::decode(bm_args);
    bm_configs.emplace_back(bm_config);
  }
}

std::vector<BenchmarkConfig> BenchmarkFactory::create_benchmark_matrix(YAML::Node& config_args,
                                                                       YAML::Node& matrix_args) {
  if (!matrix_args.IsMap()) {
    spdlog::critical("'matrix' must be a YAML map.");
    utils::crash_exit();
  }

  auto matrix = std::vector<BenchmarkConfig>{};
  auto matrix_arg_names = std::set<std::string>{};

  std::function<void(const YAML::iterator&, YAML::Node&)> create_matrix = [&](const YAML::iterator& node_iterator,
                                                                              YAML::Node& current_config) {
    YAML::Node current_values = node_iterator->second;
    if (node_iterator == matrix_args.end() || current_values.IsNull()) {
      // End of matrix recursion.
      // We need to copy here to keep the tags clean in the YAML.
      // Otherwise, everything is 'visited' after the first iteration and decoding fails.
      YAML::Node clean_config = YAML::Clone(current_config);

      auto benchmark_config = BenchmarkConfig::decode(clean_config);
      benchmark_config.matrix_args = {matrix_arg_names.begin(), matrix_arg_names.end()};

      matrix.emplace_back(benchmark_config);
      return;
    }

    if (!current_values.IsSequence()) {
      spdlog::critical("Matrix entries must be a YAML sequence, i.e., [a, b, c].");
      utils::crash_exit();
    }

    const auto arg_name = node_iterator->first.as<std::string>();
    matrix_arg_names.insert(arg_name);
    YAML::iterator next_node = node_iterator;
    next_node++;
    for (YAML::Node value : current_values) {
      current_config[arg_name] = value;
      create_matrix(next_node, current_config);
    }
  };

  YAML::Node base_config = YAML::Clone(config_args);
  create_matrix(matrix_args.begin(), base_config);
  return matrix;
}

std::vector<YAML::Node> BenchmarkFactory::get_config_files(const std::filesystem::path& config_file_path) {
  std::vector<std::filesystem::path> config_files{};
  if (std::filesystem::is_directory(config_file_path)) {
    for (const std::filesystem::path& config_file : std::filesystem::recursive_directory_iterator(config_file_path)) {
      if (config_file.extension() == CONFIG_FILE_EXTENSION) {
        config_files.push_back(config_file);
      }
    }
  } else {
    config_files.push_back(config_file_path);
  }

  if (config_files.empty()) {
    spdlog::critical("Benchmark config path {} must contain at least one config file.", config_file_path.string());
    utils::crash_exit();
  }
  std::vector<YAML::Node> yaml_configs{};
  try {
    for (const auto& config_file : config_files) {
      auto config = YAML::LoadFile(config_file);

      // Add the config file name to the config.
      for (auto it = config.begin(); it != config.end(); ++it) {
        const auto bm_group_name = it->first;
        // The config file must be added to different positions in single and parallel benchmarks.

        // Single Benchmark
        if (!config[bm_group_name]["parallel_benchmark"].IsDefined()) {
          config[bm_group_name]["args"]["config_file"] = config_file.string();
          continue;
        }

        // Parallel Benchmark
        auto workload_it = config[bm_group_name]["parallel_benchmark"].begin();
        const auto workload_end = config[bm_group_name]["parallel_benchmark"].end();
        for (; workload_it != workload_end; ++workload_it) {
          const auto& workload_name = workload_it->first;
          config[bm_group_name]["parallel_benchmark"][workload_name]["args"]["config_file"] = config_file.string();
        }
      }

      yaml_configs.emplace_back(config);
    }
  } catch (const YAML::ParserException& e1) {
    spdlog::critical("Exception during config parsing: {}", e1.msg);
    utils::crash_exit();
  } catch (const YAML::BadFile& e2) {
    spdlog::critical("Exception during config parsing: {}", e2.msg);
    utils::crash_exit();
  }
  return yaml_configs;
}

}  // namespace mema
