#include <spdlog/spdlog.h>

#include <map>

#include "CLI11.hpp"
#include "benchmark_suite.hpp"
#include "numa.hpp"

namespace {

std::string empty_directory(const std::string& path) {
  if (!std::filesystem::is_empty(path)) {
    return "PMem benchmark directory '" + path + "' must be empty.";
  }
  return "";
}

}  // namespace

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

  // Define NUMA nodes to pin tasks to. This takes a list of nodes, e.g., --numa=0,1
  NumaNodeIDs numa_task_nodes;
  auto numa_task_opt = app.add_option("--numa_task", numa_task_nodes,
                                      "Comma separated list of NUMA nodes to pin tasks to, e.g., --numa=0,1 "
                                      "(default: determined from PMem directory)")
                           ->delimiter(',')
                           ->required()
                           ->expected(1, 10);

  // Path to PMem directory
  std::filesystem::path pmem_directory;
  auto path_opt = app.add_option("-p,--path", pmem_directory,
                                 "Path to empty memory directory (e.g., PMem directory) in which to perform the "
                                 "benchmarks, e.g., /mnt/pmem1/mema")
                      ->default_str("")
                      ->check(CLI::ExistingDirectory)
                      ->check(empty_directory);

  // Flag if DRAM should be used
  bool use_dram;
  auto dram_flg = app.add_flag("--dram", use_dram, "Set this flag to run benchmarks in DRAM")->default_val(true);

  // Do not allow path to be set if dram is set
  dram_flg->excludes(path_opt);
  // Do not allow dram flag to be set if path is set
  path_opt->excludes(dram_flg);

  try {
    app.parse(argc, argv);
    if (path_opt->empty() && dram_flg->empty()) {
      throw CLI::RequiredError("Must specify either --path or --dram");
    }
  } catch (const CLI::ParseError& e) {
    app.failure_message(CLI::FailureMessage::help);
    return app.exit(e);
  }

  // TODO(MW) remove this while getting rid of the binary memory type switch.
  // For now, we only support DRAM mode.
  if (!use_dram) {
    spdlog::error("PMem was chosen. We currently only support DRAM benchmarks.");
    exit(1);
  }

  if (be_verbose) {
    spdlog::set_level(spdlog::level::debug);
  }

  // Make sure that the benchmarks are NUMA-aware. Setting this in the main thread will inherit to all child threads.
  init_numa(numa_task_nodes);

  // Run the actual benchmarks after parsing and validating them.
  const std::string run_location = use_dram ? "DRAM" : pmem_directory.string();
  spdlog::info("Running benchmarks on '{}' with config(s) from '{}'.", run_location, config_file.string());
  spdlog::info("Writing results to '{}'.", result_path.string());

  try {
    BenchmarkSuite::run_benchmarks({pmem_directory, config_file, result_path, !use_dram});
  } catch (const PermaException& e) {
    // Clean up files before exiting
    if (!use_dram) {
      std::filesystem::remove_all(pmem_directory / "*");
    }
    throw e;
  }

  if (!use_dram) {
    std::filesystem::remove_all(pmem_directory / "*");
  }
  return 0;
}
