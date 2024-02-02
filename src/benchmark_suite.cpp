#include "benchmark_suite.hpp"

#include <spdlog/spdlog.h>
#include <unistd.h>

#include <fstream>
#include <iostream>
#include <json.hpp>

#include "benchmark_config.hpp"
#include "benchmark_factory.hpp"
#include "utils.hpp"

namespace {

nlohmann::json single_results_to_json(const mema::SingleBenchmark& bm, const nlohmann::json& bm_results,
                                      const std::string& nt_stores_instruction_set, const std::string& git_hash,
                                      const std::string& compiler, const std::string& hostname) {
  return {{"bm_name", bm.benchmark_name()},
          {"bm_type", bm.benchmark_type_as_str()},
          {"matrix_args", bm.get_benchmark_configs()[0].matrix_args},
          {"benchmarks", bm_results},
          {"nt_stores_instruction_set", nt_stores_instruction_set},
          {"git_hash", git_hash},
          {"compiler", compiler},
          {"hostname", hostname}};
}

nlohmann::json parallel_results_to_json(const mema::ParallelBenchmark& bm, const nlohmann::json& bm_results,
                                        const std::string& nt_stores_instruction_set, const std::string& git_hash,
                                        const std::string& compiler, const std::string& hostname) {
  return {{"bm_name", bm.benchmark_name()},
          {"sub_bm_names", {bm.get_benchmark_name_one(), bm.get_benchmark_name_two()}},
          {"bm_type", bm.benchmark_type_as_str()},
          {"matrix_args",
           {{bm.get_benchmark_name_one(), bm.get_benchmark_configs()[0].matrix_args},
            {bm.get_benchmark_name_two(), bm.get_benchmark_configs()[1].matrix_args}}},
          {"benchmarks", bm_results},
          {"nt_stores_instruction_set", nt_stores_instruction_set},
          {"git_hash", git_hash},
          {"compiler", compiler},
          {"hostname", hostname}};
}

nlohmann::json benchmark_results_to_json(const mema::Benchmark& bm, const nlohmann::json& bm_results) {
  auto nt_stores_instruction_set = std::string{};
#if defined USE_AVX_512
  nt_stores_instruction_set = "avx-512";
#elif defined USE_AVX_2
  nt_stores_instruction_set = "avx-2";
#else
  nt_stores_instruction_set = "none";
#endif

  // The following call of git describe will never find a tag because of --match="no-match^". Since --always is set, it
  // will return the hash of the current commit instead.
  const auto pipe =
      std::shared_ptr<FILE>(popen("git describe --match=\"no-match^\" --always --abbrev=40 --dirty", "r"), pclose);
  if (!pipe) {
    spdlog::critical("Failed to get git hash.");
    mema::utils::crash_exit();
  }
  // 60 characters is large enough for a git commit, even if the repository is dirty.
  auto git_hash_buffer = std::array<char, 60>{};
  auto git_hash = std::string{};
  if (fgets(git_hash_buffer.data(), 60, pipe.get()) == NULL) {
    spdlog::critical("Failed to get git hash.");
    mema::utils::crash_exit();
  }
  git_hash = std::string{git_hash_buffer.data()};
  // Remove newline character.
  git_hash.pop_back();

  static auto compiler = std::string{};
  if (compiler.empty()) {
    auto stream = std::stringstream{};
#if defined(__clang__)
    stream << "clang " << __clang_major__ << "." << __clang_minor__ << "." << __clang_patchlevel__;
#elif defined(__GNUC__)
    stream << "gcc " << __GNUC__ << "." << __GNUC_MINOR__;
#else
    stream << "unknown";
#endif
    compiler = stream.str();
  }

  static auto hostname = std::string{};
  if (hostname.empty()) {
    auto hostname_buffer = std::array<char, 1024>{};
    gethostname(hostname_buffer.data(), sizeof(hostname_buffer));
    hostname = std::string{hostname_buffer.data()};
  }

  if (bm.get_benchmark_type() == mema::BenchmarkType::Single) {
    return single_results_to_json(dynamic_cast<const mema::SingleBenchmark&>(bm), bm_results, nt_stores_instruction_set,
                                  git_hash, compiler, hostname);
  } else if (bm.get_benchmark_type() == mema::BenchmarkType::Parallel) {
    return parallel_results_to_json(dynamic_cast<const mema::ParallelBenchmark&>(bm), bm_results,
                                    nt_stores_instruction_set, git_hash, compiler, hostname);
  } else {
    return {{"bm_name", bm.benchmark_name()},
            {"bm_type", bm.benchmark_type_as_str()},
            {"benchmarks", bm_results},
            {"nt_stores_instruction_set", nt_stores_instruction_set},
            {"git_hash", git_hash},
            {"compiler", compiler},
            {"hostname", hostname}};
  }
}

void print_bm_information(const mema::Benchmark& bm) {
  if (bm.get_benchmark_type() == mema::BenchmarkType::Single) {
    spdlog::info("Running single benchmark {} with matrix args {}", bm.benchmark_name(),
                 nlohmann::json(bm.get_benchmark_configs()[0].matrix_args).dump());
  } else if (bm.get_benchmark_type() == mema::BenchmarkType::Parallel) {
    const auto& benchmark = dynamic_cast<const mema::ParallelBenchmark&>(bm);
    spdlog::info("Running parallel benchmark {} with sub benchmarks {} and {}.", benchmark.benchmark_name(),
                 benchmark.get_benchmark_name_one(), benchmark.get_benchmark_name_two());
  } else {
    // This should never happen
    spdlog::critical("Unknown benchmark type: {}", bm.get_benchmark_type());
    mema::utils::crash_exit();
  }
}

void print_summary(nlohmann::json result, const std::string& bm_name, const bool parallel = false) {
  if (result["benchmarks"].size() == 0) {
    spdlog::critical("No results found for benchmark '{}'.", bm_name);
    mema::utils::crash_exit();
  }

  auto prefix = std::string{};

  if (parallel) {
    prefix = "\t";

    for (auto& bm_result : result["benchmarks"]) {
      bm_result["config"] = bm_result["config"][bm_name];
      bm_result["results"] = bm_result["results"][bm_name]["results"];
    }
  }

  if (result["benchmarks"][0]["config"]["exec_mode"] == "custom") {
    auto avg_latency = double{0};
    auto min_latency = double{std::numeric_limits<double>::max()};
    auto min_config = nlohmann::json{};
    auto max_latency = double{std::numeric_limits<double>::min()};
    auto max_config = nlohmann::json{};

    for (const auto& bm_result : result["benchmarks"]) {
      if (bm_result["config"]["exec_mode"] != "custom") {
        spdlog::critical("A single benchmark can not contain mixed execution modes.");
        mema::utils::crash_exit();
      }

      const auto latency = bm_result["results"]["latency"]["avg"].get<double>();
      avg_latency += latency;
      if (latency < min_latency) {
        min_latency = latency;
        min_config = bm_result["config"];
      }
      if (latency > max_latency) {
        max_latency = latency;
        max_config = bm_result["config"];
      }
    }

    avg_latency /= result["benchmarks"].size();
    spdlog::info("{}:\tavg_latency (ns): {}\tmin_latency (ns): {}\tmax_latency (ns): {}", bm_name, avg_latency,
                 min_latency, max_latency);
    spdlog::info("{}min_latency config: {}", prefix, min_config.dump());
    spdlog::info("{}max_latency config: {}", prefix, max_config.dump());
  } else if (result["benchmarks"][0]["config"]["exec_mode"] == "sequential" ||
             result["benchmarks"][0]["config"]["exec_mode"] == "random") {
    auto avg_bandwidth = double{0};
    auto min_bandwidth = double{std::numeric_limits<double>::max()};
    auto min_config = nlohmann::json{};
    auto max_bandwidth = double{std::numeric_limits<double>::min()};
    auto max_config = nlohmann::json{};

    for (const auto& bm_result : result["benchmarks"]) {
      if (!(result["benchmarks"][0]["config"]["exec_mode"] == "sequential" ||
            result["benchmarks"][0]["config"]["exec_mode"] == "random")) {
        spdlog::critical("A single benchmark can not contain mixed execution modes.");
        mema::utils::crash_exit();
      }

      const auto bandwidth = bm_result["results"]["bandwidth"].get<double>();
      avg_bandwidth += bandwidth;
      if (bandwidth < min_bandwidth) {
        min_bandwidth = bandwidth;
        min_config = bm_result["config"];
      }
      if (bandwidth > max_bandwidth) {
        max_bandwidth = bandwidth;
        max_config = bm_result["config"];
      }
    }

    // TODO(anyone): use 1 GB for bandwidth measurements in the code base.
    // 1 GiB / 1 GB
    min_bandwidth *= mema::utils::ONE_GB / 1e9;
    max_bandwidth *= mema::utils::ONE_GB / 1e9;
    avg_bandwidth *= mema::utils::ONE_GB / 1e9;
    avg_bandwidth /= result["benchmarks"].size();

    spdlog::info("{}{}:\tavg_bandwidth (GB/s): {}\tmin_bandwidth (GB/s): {}\tmax_bandwidth (GB/s): {}", prefix, bm_name,
                 avg_bandwidth, min_bandwidth, max_bandwidth);
    spdlog::info("{}min_bandwidth config: {}", prefix, min_config.dump());
    spdlog::info("{}max_bandwidth config: {}", prefix, max_config.dump());
  } else {
    spdlog::critical("Unknown execution mode: {}", result["benchmarks"][0]["config"]["exec_mode"]);
    mema::utils::crash_exit();
  }
}

void print_summarys(const nlohmann::json& all_results) {
  spdlog::info("Summary:");
  for (const auto& result : all_results) {
    const auto& bm_name = result["bm_name"].get<std::string>();
    if (result["bm_type"] == "single") {
      print_summary(result, bm_name);
    } else if (result["bm_type"] == "parallel") {
      spdlog::info("{}:", bm_name);
      const auto& sub_bm_names = result["sub_bm_names"].get<std::vector<std::string>>();
      for (const auto& sub_bm_name : sub_bm_names) {
        print_summary(result, sub_bm_name, true);
      }
    } else {
      spdlog::critical("Unknown benchmark type: {}", result["bm_type"]);
      mema::utils::crash_exit();
    }
  }
}

}  // namespace

namespace mema {

void BenchmarkSuite::run_benchmarks(const MemaOptions& options) {
  auto yaml_configs = BenchmarkFactory::get_config_files(options.config_file);

  // -------------------------------------------------------------------------------------------------------------------
  // Create single and parallel benchmarks

  std::vector<SingleBenchmark> single_benchmarks = BenchmarkFactory::create_single_benchmarks(yaml_configs);
  spdlog::info("Found {} single benchmark{}.", single_benchmarks.size(), single_benchmarks.size() != 1 ? "s" : "");

  std::vector<ParallelBenchmark> parallel_benchmarks = BenchmarkFactory::create_parallel_benchmarks(yaml_configs);
  spdlog::info("Found {} parallel benchmark{}.", parallel_benchmarks.size(),
               parallel_benchmarks.size() != 1 ? "s" : "");

  auto benchmarks = std::vector<Benchmark*>{};
  benchmarks.reserve(single_benchmarks.size() + parallel_benchmarks.size());

  for (Benchmark& benchmark : single_benchmarks) {
    benchmarks.push_back(&benchmark);
  }
  if (options.only_equal_thread_counts) {
    // Lambda checks if the passed benchmark has the same thread count across all configurations.
    auto has_equal_thread_counts_across_workloads = [](auto& benchmark) {
      auto thread_count = -1;
      for (auto& config : benchmark.get_benchmark_configs()) {
        if (thread_count == -1) {
          thread_count = config.number_threads;
          continue;
        }
        if (config.number_threads != thread_count) {
          return false;
        }
      }

      return true;
    };

    auto filtered_count = 0u;
    for (Benchmark& benchmark : parallel_benchmarks) {
      if (!has_equal_thread_counts_across_workloads(benchmark)) {
        ++filtered_count;
        continue;
      }
      benchmarks.push_back(&benchmark);
    }
    spdlog::info("Filtered out {} parallel benchmark{}.", filtered_count, filtered_count != 1 ? "s" : "");
  } else {
    for (Benchmark& benchmark : parallel_benchmarks) {
      benchmarks.push_back(&benchmark);
    }
  }
  // -------------------------------------------------------------------------------------------------------------------

  const std::filesystem::path result_file = utils::create_result_file(options.result_directory, options.config_file);

  if (benchmarks.empty()) {
    spdlog::warn("No benchmarks found. Nothing to do.");
    return;
  }

  auto had_error = false;
  auto printed_info = false;
  Benchmark* previous_bm = nullptr;
  auto matrix_bm_results = nlohmann::json::array();

  const auto benchmark_count = benchmarks.size();
  for (auto bench_idx = uint64_t{0}; bench_idx < benchmark_count; ++bench_idx) {
    auto& benchmark = *benchmarks[bench_idx];
    const auto is_parallel = benchmark.get_benchmark_type() == BenchmarkType::Parallel;

    spdlog::info("Executing benchmark {0}, {1}:", bench_idx + 1, benchmark.benchmark_name());
    if (is_parallel) {
      spdlog::info("Worklaod 0: {0}", benchmark.get_benchmark_configs()[0].to_string());
      spdlog::info("Worklaod 1: {0}", benchmark.get_benchmark_configs()[1].to_string());
    } else {
      spdlog::info("{0}", benchmark.get_benchmark_configs()[0].to_string());
    }

    if (previous_bm && previous_bm->benchmark_name() != benchmark.benchmark_name()) {
      // Started new benchmark, force delete old data in case it was a matrix. If it is not a matrix, this does nothing.
      auto bm_results = benchmark_results_to_json(*previous_bm, matrix_bm_results);
      utils::write_benchmark_results(result_file, bm_results);
      matrix_bm_results = nlohmann::json::array();
      previous_bm->tear_down(/*force=*/true);
      printed_info = false;
    }

    if (!printed_info) {
      print_bm_information(benchmark);
      printed_info = true;
    }

    if (is_parallel) {
      spdlog::debug("Preparing parallel benchmark #{} with two configs: {} AND {}", benchmark_count,
                    to_string(benchmark.get_json_config(0)), to_string(benchmark.get_json_config(1)));
    } else {
      spdlog::debug("Preparing benchmark #{} with config: {}", benchmark_count,
                    to_string(benchmark.get_json_config(0)));
    }

    benchmark.generate_data();
    spdlog::debug("Finished generating data.");
    benchmark.set_up();
    spdlog::debug("Finished setting up benchmark.");
    const bool success = benchmark.run();
    spdlog::debug("Finished running benchmark.");
    previous_bm = &benchmark;

    if (!success) {
      // Encountered an error. End suite gracefully.
      had_error = true;
      break;
    }

    matrix_bm_results += benchmark.get_result_as_json();
    benchmark.tear_down(false);
    spdlog::info("Completed {0}/{1} benchmark{2}.", bench_idx + 1, benchmarks.size(), benchmarks.size() > 1 ? "s" : "");
  }

  if (!benchmarks.empty()) {
    nlohmann::json bm_results = benchmark_results_to_json(*previous_bm, matrix_bm_results);
    utils::write_benchmark_results(result_file, bm_results);
    std::filesystem::create_symlink(result_file, utils::LAST_RESULTS_FILENAME);
    previous_bm->tear_down(/*force=*/true);
  }

  if (had_error) {
    utils::crash_exit();
  }

  auto all_results = nlohmann::json{};
  auto previous_result_file = std::ifstream{result_file};
  previous_result_file >> all_results;

  if (!all_results.is_array()) {
    previous_result_file.close();
    spdlog::critical("Result file '{}' is corrupted! Content must be a valid JSON array.", result_file.string());
    utils::crash_exit();
  }

  print_summarys(all_results);
  spdlog::debug("Results:\n{}", all_results.dump(2));

  spdlog::info("Finished all benchmarks successfully.");
}

}  // namespace mema
