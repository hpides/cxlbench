
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <map>

#include "CLI11.hpp"
#include "benchmark_suite.hpp"
#include "numa.hpp"
#include "utils.hpp"

using namespace mema;  // NOLINT - [build/namespaces] Linter doesn't like using-directives

constexpr auto DEFAULT_WORKLOAD_PATH = "workloads";
constexpr auto DEFAULT_RESULT_PATH = "results";

int main(int argc, char** argv) {
  CLI::App app{"MemA-Bench: Benchmark your Memory"};

  // Delete symlink to last result file if it exists
  std::remove(utils::LAST_RESULTS_FILENAME);

  // Set verbosity
  bool be_verbose;
  app.add_flag("-v,--verbose", be_verbose, "Set true to log additional runtime information.")->default_val(false);

  // Define command line args
  std::filesystem::path config_path = std::filesystem::current_path() / DEFAULT_WORKLOAD_PATH;
  app.add_option("-c,--config", config_path,
                 "Path to the benchmark config YAML file(s) (default: " + std::string{DEFAULT_WORKLOAD_PATH} + ")");

  // Define result directory
  std::filesystem::path result_path = std::filesystem::current_path() / DEFAULT_RESULT_PATH;
  app.add_option("-r,--results", result_path, "Path to the result directory (default: " + result_path.string() + ")");

  // Parrallel workloads: only execute configurations in which the thread count is equal across workloads.
  auto only_equal_thread_counts = false;
  app.add_flag("-e, --equal_thread_count", only_equal_thread_counts,
               "Set true to only execute parallel benchmarks with an equal number of threads accross the workloads.")
      ->default_val(false);

  try {
    app.parse(argc, argv);
    spdlog::debug("Parsed command line arguments.");
  } catch (const CLI::ParseError& e) {
    app.failure_message(CLI::FailureMessage::help);
    return app.exit(e);
  }

  // Set up logging
  //--------------------------------------------------------------------------------------------------------------------

  // stdout logger
  auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
  // Logs with level >= trace are shown in this sink
  console_sink->set_level(spdlog::level::trace);

  // file logger
  auto file_name = utils::get_file_name_from_path(config_path, "log");
  auto file_path = std::filesystem::path(result_path) / file_name;
  auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(file_path, false);
  // Logs with level >= trace are shown in this sink
  file_sink->set_level(spdlog::level::trace);

  auto combined_logger =
      std::make_shared<spdlog::logger>("multi_sink", spdlog::sinks_init_list({console_sink, file_sink}));
  spdlog::set_default_logger(combined_logger);
  spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");
  spdlog::flush_on(spdlog::level::trace);

#ifdef NDEBUG
  spdlog::set_level(spdlog::level::info);
#else
  spdlog::set_level(spdlog::level::debug);
#endif

  if (be_verbose) {
    spdlog::set_level(spdlog::level::debug);
  }

  // Verify Page Size
  const auto page_size = getpagesize();
  if (utils::PAGE_SIZE != page_size) {
    spdlog::critical("System has a page size {}, but the benchmark tool assumes {}.", page_size, utils::PAGE_SIZE);
    utils::crash_exit();
  }

  // Run the actual benchmarks after parsing and validating them.
  //--------------------------------------------------------------------------------------------------------------------
  spdlog::info("Running benchmarks with config(s) from '{}'.", config_path.string());
  if (!std::filesystem::exists(config_path)) {
    spdlog::critical(
        "Config path {} does not exist. Make sure you set up the benchmark configurations correctly. Feel free to use "
        "../scripts/reset_workload.sh",
        config_path.string());
    utils::crash_exit();
  }
  spdlog::info("Writing results to '{}'.", result_path.string());

  const auto numa_node_count = init_numa();
  spdlog::info("Number of NUMA nodes in system: {}", numa_node_count);

  BenchmarkSuite::run_benchmarks({config_path, result_path, only_equal_thread_counts});
  return 0;
}
