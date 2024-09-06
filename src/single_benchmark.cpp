#include "single_benchmark.hpp"

#include <csignal>
// TODO(MW) remove
#include <iostream>

#include "numa.hpp"

namespace {

volatile sig_atomic_t thread_error;
void thread_error_handler(int) { thread_error = 1; }

}  // namespace

namespace cxlbench {

void SingleBenchmark::log_config() { spdlog::info("Single workload: {0}", configs_[0].to_string()); }

void SingleBenchmark::debug_log_json_config(size_t benchmark_idx) {
  spdlog::debug("Preparing single benchmark #{} with config: {}", benchmark_idx, to_string(get_json_config(0)));
}
void SingleBenchmark::log_information() {
  spdlog::info("Running single benchmark {} with matrix args {}", benchmark_name(),
               nlohmann::json(get_benchmark_configs()[0].matrix_args).dump());
}

bool SingleBenchmark::run() {
  signal(SIGSEGV, thread_error_handler);

  utils::clear_caches();
  const BenchmarkConfig& config = configs_[0];
  auto& thread_pool = thread_pools_[0];
  for (size_t thread_index = 0; thread_index < config.number_threads; thread_index++) {
    thread_pool.emplace_back(run_in_thread, &thread_configs_[0][thread_index], std::ref(config));
  }

  // wait for all threads
  for (std::thread& thread : thread_pool) {
    if (thread_error) {
      utils::print_segfault_error();
      return false;
    }
    thread.join();
  }

  return true;
}

void SingleBenchmark::generate_data() {
  if (!memory_region_sets_.empty()) {
    spdlog::critical("generate_data() called more than once for the same benchmark.");
    utils::crash_exit();
  }
  memory_region_sets_.resize(1);
  memory_region_sets_[0] = prepare_data(configs_[0]);
}

void SingleBenchmark::set_up() {
  thread_pools_.resize(1);
  thread_configs_.resize(1);
  single_set_up(configs_[0], memory_region_sets_[0], executions_[0].get(), results_[0].get(), &thread_pools_[0],
                &thread_configs_[0]);
}

void SingleBenchmark::verify() {
  if (configs_[0].is_memory_management_op()) {
    spdlog::info("Skipping verification for memory management operation.");
    return;
  }

  verify_page_locations(memory_region_sets_[0], configs_[0].memory_regions, 0);
}

nlohmann::json SingleBenchmark::get_result_as_json() {
  nlohmann::json result;
  result["config"] = get_json_config(0);
  result.update(results_[0]->get_result_as_json());
  return result;
}

SingleBenchmark::SingleBenchmark(const std::string& benchmark_name, const BenchmarkConfig& config,
                                 std::vector<std::unique_ptr<BenchmarkExecution>>&& executions,
                                 std::vector<std::unique_ptr<BenchmarkResult>>&& results)
    : Benchmark(benchmark_name, BenchmarkType::Single, std::vector<BenchmarkConfig>{config}, std::move(executions),
                std::move(results)) {}

}  // namespace cxlbench
