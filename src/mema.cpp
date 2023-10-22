
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <map>

#include "CLI11.hpp"
#include "benchmark_suite.hpp"
#include "numa.hpp"

using namespace mema;  // NOLINT - [build/namespaces] Linter doesn't like using-directives

constexpr auto DEFAULT_WORKLOAD_PATH = "workloads";
constexpr auto DEFAULT_RESULT_PATH = "results";
constexpr auto LOG_PATH = "logs";

int main(int argc, char** argv) {
  CLI::App app{"MemA-Bench: Benchmark your Memory"};

  // Delete symlink to last result file if it exists
  std::remove(utils::LAST_RESULTS_FILENAME);

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

  try {
    app.parse(argc, argv);
    spdlog::debug("Parsed command line arguments.");
  } catch (const CLI::ParseError& e) {
    app.failure_message(CLI::FailureMessage::help);
    return app.exit(e);
  }

  // Set up logging

  // stdout logger
  auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
  // Logs with level >= trace are shown in this sink
  console_sink->set_level(spdlog::level::trace);

  // file logger
  auto file_path = std::filesystem::path(LOG_PATH) / (utils::get_time_string() + ".log");
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

  // Run the actual benchmarks after parsing and validating them.
  spdlog::info("Running benchmarks with config(s) from '{}'.", config_file.string());
  spdlog::info("Writing results to '{}'.", result_path.string());

  const auto numa_node_count = init_numa();
  spdlog::info("Number of NUMA nodes in system: {}", numa_node_count);

  // Lock the whole memory associated with the calling process. This prevents the OS from swapping even if it's
  // activated. Swapping out memory would invalidate the results of the benchmarks.
  const auto ret = mlockall(MCL_CURRENT | MCL_FUTURE);
  if (ret != 0) {
    spdlog::critical("Failed to lock memory.");
    utils::crash_exit();
  }

  BenchmarkSuite::run_benchmarks({config_file, result_path});
  return 0;
}
