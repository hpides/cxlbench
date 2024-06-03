#pragma once

#include <hdr_histogram.h>

#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <map>
#include <random>
#include <utility>
#include <vector>

#include "benchmark_config.hpp"
#include "io_operation.hpp"
#include "json.hpp"
#include "utils.hpp"

namespace mema {

enum class BenchmarkType : uint8_t { Single, Parallel };

struct BenchmarkEnums {
  static const std::unordered_map<std::string, BenchmarkType> str_to_benchmark_type;
};

struct ExecutionDuration {
  std::chrono::steady_clock::time_point begin;
  std::chrono::steady_clock::time_point end;

  std::chrono::steady_clock::duration duration() const { return end - begin; }
};

struct BenchmarkExecution {
  // Owning instance for thread synchronization
  std::mutex generation_lock{};
  std::condition_variable generation_done{};
  uint16_t threads_remaining;
  std::atomic<uint64_t> io_position = 0;

  // For custom operations, we don't have chunks but only simulate them by running chunk-sized blocks.
  // This is a *signed* integer, as our atomic -= operations my go below 0.
  std::atomic<int64_t> num_custom_chunks_remaining = 0;

  // The main list of all IO operations to steal work from
  std::vector<IoOperation> io_operations;
};

struct ThreadRunConfig {
  // Partition start addresses
  char* start_addr;
  char* secondary_start_addr;

  const uint64_t partition_size;
  const uint64_t secondary_partition_size;
  const uint64_t thread_count_per_partition;
  const uint64_t thread_idx;
  const uint64_t ops_count_per_chunk;
  const uint64_t chunk_count;
  const BenchmarkConfig& config;

  BenchmarkExecution* execution;

  // Pointers to store performance data in.
  uint64_t* total_operation_size;
  ExecutionDuration* total_operation_duration;
  std::vector<uint64_t>* custom_op_latencies;

  ThreadRunConfig(char* partition_start_addr, char* secondary_partition_start_addr, const uint64_t partition_size,
                  const uint64_t secondary_partition_size, const uint64_t thread_count_per_partition,
                  const uint64_t thread_idx, const uint64_t ops_count_per_chunk, const uint64_t chunk_count,
                  const BenchmarkConfig& config, BenchmarkExecution* execution,
                  ExecutionDuration* total_operation_duration, uint64_t* total_operation_size,
                  std::vector<uint64_t>* custom_op_latencies)
      : start_addr{partition_start_addr},
        secondary_start_addr{secondary_partition_start_addr},
        partition_size{partition_size},
        secondary_partition_size{secondary_partition_size},
        thread_count_per_partition{thread_count_per_partition},
        thread_idx{thread_idx},
        ops_count_per_chunk{ops_count_per_chunk},
        chunk_count{chunk_count},
        config{config},
        execution{execution},
        total_operation_duration{total_operation_duration},
        total_operation_size{total_operation_size},
        custom_op_latencies{custom_op_latencies} {}
};

struct BenchmarkResult {
  explicit BenchmarkResult(BenchmarkConfig config);
  ~BenchmarkResult();

  nlohmann::json get_result_as_json() const;
  nlohmann::json get_custom_results_as_json() const;

  // Result vectors for raw operation workloads
  std::vector<uint64_t> total_operation_sizes;
  std::vector<ExecutionDuration> total_operation_durations;

  // Result vectors for custom operation workloads
  std::vector<std::vector<uint64_t>> custom_operation_latencies;

  hdr_histogram* latency_hdr = nullptr;
  const BenchmarkConfig config;
};

class Benchmark {
 public:
  Benchmark(std::string benchmark_name, BenchmarkType benchmark_type, std::vector<BenchmarkConfig> configs,
            std::vector<std::unique_ptr<BenchmarkExecution>> executions,
            std::vector<std::unique_ptr<BenchmarkResult>> results)
      : benchmark_name_{std::move(benchmark_name)},
        benchmark_type_{benchmark_type},
        configs_{std::move(configs)},
        executions_{std::move(executions)},
        results_{std::move(results)} {}

  Benchmark(Benchmark&& other) = default;
  Benchmark(const Benchmark& other) = delete;
  Benchmark& operator=(const Benchmark& other) = delete;
  Benchmark& operator=(Benchmark&& other) = delete;

  /**
   * Main run method which executes the benchmark. `set_up()` should be called before this.
   * Return true if benchmark ran successfully, false if an error was encountered.
   */
  virtual bool run() = 0;

  /**
   * Generates the data needed for the benchmark.
   * This is probably the first method to be called so that a virtual
   * address space is available to generate the IO addresses.
   */
  virtual void generate_data() = 0;

  /** Create all the IO addresses ahead of time to avoid unnecessary ops during the actual benchmark. Prepare worker
   * thread configurations. */
  virtual void set_up() = 0;

  virtual void verify() = 0;

  /** Return the results as a JSON to be exported to the user and visualization. */
  virtual nlohmann::json get_result_as_json() = 0;

  /** Clean up after te benchmark */
  void tear_down(bool force);

  /** Return the name of the benchmark. */
  const std::string& benchmark_name() const;

  /** Return the type of the benchmark. */
  std::string benchmark_type_as_str() const;

  // Return the benchmark Type.
  BenchmarkType get_benchmark_type() const;

  // Returns the memory regions (inner vector) for the different workloads (outer vector).
  const std::vector<MemoryRegions>& get_memory_regions() const;

  const std::vector<BenchmarkConfig>& get_benchmark_configs() const;
  const std::vector<std::vector<ThreadRunConfig>>& get_thread_configs() const;
  const std::vector<std::unique_ptr<BenchmarkResult>>& get_benchmark_results() const;

  nlohmann::json get_json_config(uint8_t config_index);

 protected:
  static void single_set_up(const BenchmarkConfig& config, MemoryRegions& memory_regions, BenchmarkExecution* execution,
                            BenchmarkResult* result, std::vector<std::thread>* pool,
                            std::vector<ThreadRunConfig>* thread_config);

  // Pepares the memory regions and returns the start pointer. A start address is nullptr if the corresponding memory
  // region has a size of 0.
  static MemoryRegions prepare_data(const BenchmarkConfig& config);

  // Prepares the memory region with pages interleaved accross the given NUMA nodes.
  static char* prepare_interleaved_data(const MemoryRegionDefinition& region, bool prepare_read_data);

  // Prepares the memory region with two partitions each being located on a different NUMA nodes.
  static char* prepare_partitioned_data(const MemoryRegionDefinition& region, bool prepare_read_data);

  // Verifies the page locations
  static void verify_page_locations(const MemoryRegions& memory_regions,
                                    const MemoryRegionDefinitions& region_definitions);

  static void run_custom_ops_in_thread(ThreadRunConfig* thread_config, const BenchmarkConfig& config);
  static void run_in_thread(ThreadRunConfig* thread_config, const BenchmarkConfig& config);

  static uint64_t run_fixed_sized_benchmark(std::vector<IoOperation>* vector, std::atomic<uint64_t>* io_position);
  static uint64_t run_duration_based_benchmark(std::vector<IoOperation>* io_operations,
                                               std::atomic<uint64_t>* io_position,
                                               std::chrono::steady_clock::time_point execution_end);

  const std::string benchmark_name_;

  const BenchmarkType benchmark_type_;

  // Data for different workloads.
  std::vector<MemoryRegions> memory_regions_;
  const std::vector<BenchmarkConfig> configs_;
  std::vector<std::unique_ptr<BenchmarkResult>> results_;
  std::vector<std::unique_ptr<BenchmarkExecution>> executions_;
  std::vector<std::vector<ThreadRunConfig>> thread_configs_;
  std::vector<std::vector<std::thread>> thread_pools_;
};

}  // namespace mema
