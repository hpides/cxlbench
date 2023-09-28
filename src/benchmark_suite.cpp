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
#if defined NT_STORES_AVX_512
  nt_stores_instruction_set = "avx-512";
#elif defined NT_STORES_AVX_2
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

}  // namespace

namespace mema {

void BenchmarkSuite::run_benchmarks(const MemaOptions& options) {
  std::vector<YAML::Node> configs = BenchmarkFactory::get_config_files(options.config_file);
  nlohmann::json results = nlohmann::json::array();

  // Create single benchmarks
  std::vector<SingleBenchmark> single_benchmarks = BenchmarkFactory::create_single_benchmarks(configs);
  spdlog::info("Found {} single benchmark{}.", single_benchmarks.size(), single_benchmarks.size() != 1 ? "s" : "");

  // Create parallel benchmarks
  std::vector<ParallelBenchmark> parallel_benchmarks = BenchmarkFactory::create_parallel_benchmarks(configs);
  spdlog::info("Found {} parallel benchmark{}.", parallel_benchmarks.size(),
               parallel_benchmarks.size() != 1 ? "s" : "");

  auto benchmarks = std::vector<Benchmark*>{};
  benchmarks.reserve(single_benchmarks.size() + parallel_benchmarks.size());

  for (Benchmark& benchmark : single_benchmarks) {
    benchmarks.push_back(&benchmark);
  }
  for (Benchmark& benchmark : parallel_benchmarks) {
    benchmarks.push_back(&benchmark);
  }

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
  for (size_t bench_idx = 0; bench_idx < benchmark_count; ++bench_idx) {
    auto& benchmark = *benchmarks[bench_idx];
    spdlog::info("Executing benchmark {0}, {1}:", bench_idx + 1, benchmark.benchmark_name());
    spdlog::info("{0}", benchmark.get_benchmark_configs()[0].to_string());
    if (previous_bm && previous_bm->benchmark_name() != benchmark.benchmark_name()) {
      // Started new benchmark, force delete old data in case it was a matrix.
      // If it is not a matrix, this does nothing.
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

    if (benchmark.get_benchmark_type() == BenchmarkType::Parallel) {
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
    char hostname[1024] = "";
    gethostname(hostname, sizeof(hostname));
    auto ss = std::stringstream{};
    ss << "scp " << hostname << ":" << result_file << " .";
    std::cout << ss.str() << std::endl;
  }

  if (had_error) {
    utils::crash_exit();
  }

  std::stringstream buffer;
  std::ifstream file_stream{result_file};
  buffer << file_stream.rdbuf();
  spdlog::debug("Results:\n{}", buffer.str());

  spdlog::info("Finished all benchmarks successfully.");
}

}  // namespace mema
