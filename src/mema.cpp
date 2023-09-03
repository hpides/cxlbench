#include <spdlog/spdlog.h>

#include <map>

#include "CLI11.hpp"
#include "benchmark_suite.hpp"
#include "numa.hpp"

using namespace mema;  // NOLINT - [build/namespaces] Linter doesn't like using-directives

constexpr auto DEFAULT_WORKLOAD_PATH = "workloads";
constexpr auto DEFAULT_RESULT_PATH = "results";

int main(int argc, char** argv) {
#ifdef NDEBUG
  spdlog::set_level(spdlog::level::info);
#else
  spdlog::set_level(spdlog::level::debug);
#endif

  CLI::App app{"MemA-Bench: Benchmark your Memory"};

  // Set verbosity
  bool be_verbose;
  app.add_flag("-v,--verbose", be_verbose, "Set true to log additional runtime information.")->default_val(false);

  // Define command line args
  std::filesystem::path config_file = std::filesystem::current_path() / DEFAULT_WORKLOAD_PATH;
  app.add_option("-c,--config", config_file,
                 "Path to the benchmark config YAML file(s) (default: " + std::string{DEFAULT_WORKLOAD_PATH} + ")")
      ->check(CLI::ExistingPath);

  // Define result directory
  std::filesystem::path result_path = std::filesystem::current_path() / DEFAULT_RESULT_PATH;
  app.add_option("-r,--results", result_path, "Path to the result directory (default: " + result_path.string() + ")");

  if (be_verbose) {
    spdlog::set_level(spdlog::level::debug);
  }

  // Run the actual benchmarks after parsing and validating them.
  spdlog::info("Running benchmarks with config(s) from '{}'.", config_file.string());
  spdlog::info("Writing results to '{}'.", result_path.string());

  const auto numa_node_count = init_numa();
  spdlog::info("Number of NUMA nodes in system: {}", numa_node_count);

  BenchmarkSuite::run_benchmarks({config_file, result_path});
  return 0;
}
