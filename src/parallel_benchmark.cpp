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

  for (auto bm_idx = uint64_t{0}; bm_idx < configs_.size(); ++bm_idx) {
    for (auto thread_index = uint64_t{0}; thread_index < configs_[bm_idx].number_threads; thread_index++) {
      thread_pools_[bm_idx].emplace_back(&run_in_thread, &thread_configs_[bm_idx][thread_index],
                                         std::ref(configs_[bm_idx]));
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
  if (!data_.empty()) {
    spdlog::critical("generate_data() called more than once for the same benchmark.");
    utils::crash_exit();
  }
  data_.push_back(prepare_data(configs_[0], configs_[0].memory_region_size));
  data_.push_back(prepare_data(configs_[1], configs_[1].memory_region_size));
  utils::verify_memory_location(data_[0], configs_[0].memory_region_size, configs_[0].numa_memory_nodes);
  utils::verify_memory_location(data_[1], configs_[1].memory_region_size, configs_[1].numa_memory_nodes);
}

void ParallelBenchmark::set_up() {
  thread_pools_.resize(2);
  thread_configs_.resize(2);
  single_set_up(configs_[0], data_[0], executions_[0].get(), results_[0].get(), &thread_pools_[0], &thread_configs_[0]);
  single_set_up(configs_[1], data_[1], executions_[1].get(), results_[1].get(), &thread_pools_[1], &thread_configs_[1]);
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
