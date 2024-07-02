#include "parallel_benchmark.hpp"

#include <csignal>

#include "numa.hpp"

namespace {

volatile sig_atomic_t thread_error;
void thread_error_handler(int) { thread_error = 1; }

}  // namespace

namespace mema {

bool ParallelBenchmark::run() {
  signal(SIGSEGV, thread_error_handler);

  utils::clear_caches();
  for (auto workload_idx = uint64_t{0}; workload_idx < configs_.size(); ++workload_idx) {
    for (auto thread_idx = uint64_t{0}; thread_idx < configs_[workload_idx].number_threads; thread_idx++) {
      thread_pools_[workload_idx].emplace_back(&run_in_thread, &thread_configs_[workload_idx][thread_idx],
                                               std::ref(configs_[workload_idx]));
    }
  }

  // wait for all threads
  for (auto& thread_pool : thread_pools_) {
    for (auto& thread : thread_pool) {
      if (thread_error) {
        utils::print_segfault_error();
        return false;
      }
      thread.join();
    }
  }

  return true;
}

void ParallelBenchmark::generate_data() {
  if (!memory_regions_.empty()) {
    spdlog::critical("generate_data() called more than once for the same benchmark.");
    utils::crash_exit();
  }
  memory_regions_.resize(PAR_WORKLOAD_COUNT);
  for (auto workload_idx = uint64_t{0}; workload_idx < PAR_WORKLOAD_COUNT; ++workload_idx) {
    memory_regions_[workload_idx] = prepare_data(configs_[workload_idx]);
  }
}

void ParallelBenchmark::set_up() {
  thread_pools_.resize(PAR_WORKLOAD_COUNT);
  thread_configs_.resize(PAR_WORKLOAD_COUNT);
  for (auto workload_idx = uint64_t{0}; workload_idx < PAR_WORKLOAD_COUNT; ++workload_idx) {
    single_set_up(configs_[workload_idx], memory_regions_[workload_idx], executions_[workload_idx].get(),
                  results_[workload_idx].get(), &thread_pools_[workload_idx], &thread_configs_[workload_idx]);
  }
}

void ParallelBenchmark::verify() {
  for (auto workload_idx = uint32_t{0}; workload_idx < PAR_WORKLOAD_COUNT; ++workload_idx) {
    verify_page_locations(memory_regions_[workload_idx], configs_[workload_idx].memory_regions);
  }
}

nlohmann::json ParallelBenchmark::get_result_as_json() {
  nlohmann::json result;
  result["config"][benchmark_name_one_] = get_json_config(0);
  result["config"][benchmark_name_two_] = get_json_config(1);
  result["results"][benchmark_name_one_].update(results_[0]->get_result_as_json());
  result["results"][benchmark_name_two_].update(results_[1]->get_result_as_json());
  return result;
}

ParallelBenchmark::ParallelBenchmark(const std::string& benchmark_name, std::string first_benchmark_name,
                                     std::string second_benchmark_name, const BenchmarkConfig& first_config,
                                     const BenchmarkConfig& second_config,
                                     std::vector<std::unique_ptr<BenchmarkExecution>>&& executions,
                                     std::vector<std::unique_ptr<BenchmarkResult>>&& results)
    : Benchmark(benchmark_name, BenchmarkType::Parallel, std::vector<BenchmarkConfig>{first_config, second_config},
                std::move(executions), std::move(results)),
      benchmark_name_one_{std::move(first_benchmark_name)},
      benchmark_name_two_{std::move(second_benchmark_name)} {}

const std::string& ParallelBenchmark::get_benchmark_name_one() const { return benchmark_name_one_; }

const std::string& ParallelBenchmark::get_benchmark_name_two() const { return benchmark_name_two_; }

}  // namespace mema
