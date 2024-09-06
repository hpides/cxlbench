#include "benchmark.hpp"

#include <spdlog/spdlog.h>

#include <atomic>
#include <barrier>
#include <chrono>
#include <condition_variable>
#include <thread>
#include <utility>

#include "benchmark_config.hpp"
#include "fast_random.hpp"
#include "memory.hpp"
#include "numa.hpp"
#include "read_write_ops_types.hpp"
#include "threads.hpp"
#include "utils.hpp"

namespace {

template <typename T>
double calculate_standard_deviation(const std::vector<T>& values, const double average) {
  const u16 num_values = values.size();
  std::vector<double> diffs_to_avg(num_values);
  std::transform(values.begin(), values.end(), diffs_to_avg.begin(), [&](double x) { return x - average; });
  const double sq_sum = std::inner_product(diffs_to_avg.begin(), diffs_to_avg.end(), diffs_to_avg.begin(), 0.0);
  // Use N - 1 for sample variance
  const double std_dev = sqrt(sq_sum / std::max(1, num_values - 1));
  return std_dev;
}

nlohmann::json hdr_histogram_to_json(hdr_histogram* hdr) {
  nlohmann::json result;
  result["max"] = hdr_max(hdr);
  result["avg"] = hdr_mean(hdr);
  result["min"] = hdr_min(hdr);
  result["std_dev"] = hdr_stddev(hdr);
  result["median"] = hdr_value_at_percentile(hdr, 50.0);
  result["lower_quartile"] = hdr_value_at_percentile(hdr, 25.0);
  result["upper_quartile"] = hdr_value_at_percentile(hdr, 75.0);
  result["percentile_90"] = hdr_value_at_percentile(hdr, 90.0);
  result["percentile_95"] = hdr_value_at_percentile(hdr, 95.0);
  result["percentile_99"] = hdr_value_at_percentile(hdr, 99.0);
  result["percentile_999"] = hdr_value_at_percentile(hdr, 99.9);
  result["percentile_9999"] = hdr_value_at_percentile(hdr, 99.99);
  return result;
}

inline double get_bandwidth(const u64 total_data_size, const std::chrono::steady_clock::duration total_duration) {
  const double duration_in_s = static_cast<double>(total_duration.count()) / cxlbench::SECONDS_IN_NANOSECONDS;
  const double data_in_gib = static_cast<double>(total_data_size) / cxlbench::GiB;
  return data_in_gib / duration_in_s;
}

}  // namespace

namespace cxlbench {

const std::string& Benchmark::benchmark_name() const { return benchmark_name_; }

void Benchmark::log_config() {}
void Benchmark::log_information() {}
void Benchmark::debug_log_json_config(size_t benchmark_idx) {}
std::string Benchmark::benchmark_type_as_str() const {
  return utils::get_enum_as_string(BenchmarkEnums::str_to_benchmark_type, benchmark_type_);
}

void Benchmark::set_start_timestamp(const TimePointMS& start) {
  for (auto& result : results_) {
    result->start_timestamp = start;
  }
}

BenchmarkType Benchmark::get_benchmark_type() const { return benchmark_type_; }

void Benchmark::single_set_up(const BenchmarkConfig& config, MemoryRegions& memory_regions,
                              BenchmarkExecution* execution, BenchmarkResult* result,
                              std::vector<std::thread>* thread_pool, std::vector<ThreadConfig>* thread_configs) {
  const size_t total_range_op_count = config.memory_regions[0].size / config.access_size;
  const bool is_custom_execution = config.exec_mode == Mode::Custom;
  const bool is_latency_mode = config.is_latency_mode();
  const size_t num_operations =
      (config.exec_mode == Mode::Random || is_custom_execution) ? config.number_operations : total_range_op_count;
  const size_t op_count_per_thread = num_operations / config.number_threads;

  thread_pool->reserve(config.number_threads);
  thread_configs->reserve(config.number_threads);
  result->total_operation_durations.resize(config.number_threads);
  result->total_operation_sizes.resize(config.number_threads, 0);

  u64 latency_measurement_count = 0;
  if (is_custom_execution || is_latency_mode) {
    result->operation_latencies.resize(config.number_threads);

    if (config.latency_sample_frequency > 0) {
      latency_measurement_count = (op_count_per_thread / config.latency_sample_frequency) * 2;
    }
  }

  const auto thread_count = config.number_threads;
  const auto primary_region_size = config.memory_regions[0].size;
  const auto secondary_region_size = config.memory_regions[1].size;

  // Set up thread synchronization and execution parameters
  const auto& access_size =
      is_custom_execution ? CustomOp::cumulative_size(config.custom_operations) : config.access_size;
  const u64 ops_per_batch = access_size < config.min_io_batch_size ? config.min_io_batch_size / access_size : 1;

  // Add one batch for random execution and non-divisible numbers so that we perform at least number_operations ops and
  // not fewer. Adding a batch in sequential access exceeds the memory range and segfaults.
  const bool is_sequential = config.exec_mode == Mode::Sequential || config.exec_mode == Mode::Sequential_Desc;
  const size_t extra_batch = is_sequential ? 0 : (num_operations % ops_per_batch != 0);
  const size_t batch_count = (num_operations / ops_per_batch) + extra_batch;

  execution->threads_remaining = config.number_threads;
  execution->batch_position = 0;
  execution->access_batches.resize(batch_count);
  execution->num_custom_batches_remaining = static_cast<i64>(batch_count);

  // Secondary region is only used for custom operations.
  BenchAssert(config.memory_regions[1].size == 0 || config.exec_mode == Mode::Custom,
              "Secondary memory region is only supported with custom operations.");
  auto* secondary_start_addr = memory_regions[1];

  // Thread pinning preparation
  const auto is_numa_thread_pinning = (config.thread_pin_mode == ThreadPinMode::AllNumaCores ||
                                       config.thread_pin_mode == ThreadPinMode::SingleNumaCoreIncrement);
  const auto threads_pinning_cores =
      is_numa_thread_pinning ? core_ids_of_nodes(config.numa_thread_nodes) : config.thread_core_ids;

  char* primary_start_addr = (config.exec_mode == Mode::Sequential_Desc)
                                 ? memory_regions[0] + primary_region_size - access_size
                                 : memory_regions[0];

  for (u16 thread_idx = 0; thread_idx < thread_count; ++thread_idx) {
    // Reserve space for custom operation latency measurements to avoid resizing during benchmark execution.
    if (is_custom_execution) {
      result->operation_latencies[thread_idx].reserve(latency_measurement_count);
    }

    ExecutionDuration* total_op_duration = &result->total_operation_durations[thread_idx];
    u64* total_op_size = &result->total_operation_sizes[thread_idx];
    std::vector<u64>* op_latencies =
        (is_custom_execution || is_latency_mode) ? &result->operation_latencies[thread_idx] : nullptr;

    auto thread_affinity_cores = config.thread_pin_mode == ThreadPinMode::AllNumaCores
                                     ? threads_pinning_cores
                                     : CoreIDs{threads_pinning_cores[thread_idx]};

    thread_configs->emplace_back(primary_start_addr, secondary_start_addr, primary_region_size, secondary_region_size,
                                 thread_count, thread_idx, ops_per_batch, batch_count, config, thread_affinity_cores,
                                 execution, total_op_duration, total_op_size, op_latencies);
  }
}

MemoryRegions Benchmark::prepare_data(const BenchmarkConfig& config) {
  auto region_start_addresses = MemoryRegions(MEM_REGION_COUNT);
  BenchAssert(config.memory_regions.size() == MEM_REGION_COUNT, "Unexpected number of memory regions.");
  // Determines if data shall be written to the memory region so that reads can read the data.
  for (auto region_idx = u64{0}; auto& region_definition : config.memory_regions) {
    if (region_definition.size == 0 || config.is_memory_management_op()) {
      region_start_addresses[region_idx] = nullptr;
      ++region_idx;
      continue;
    }
    spdlog::info("Preparing memory region {}.", region_idx);
    switch (region_definition.placement_mode()) {
      case PagePlacementMode::NumaInterleaved:
        region_start_addresses[region_idx] = prepare_interleaved_data(region_definition, config);
        break;
      case PagePlacementMode::NumaPartitioned:
        region_start_addresses[region_idx] = prepare_partitioned_data(region_definition, config);
        break;
      case PagePlacementMode::DeviceLinear:
        region_start_addresses[region_idx] = prepare_device_data(region_definition, config);
        break;
      default:
        spdlog::critical("Data preparation mode not handled.");
        utils::crash_exit();
    }
    ++region_idx;
  }
  return region_start_addresses;
}

char* Benchmark::prepare_device_data(const MemoryRegionDefinition& region, const BenchmarkConfig& config) {
  BenchAssert(region.memory_mode() == MemoryMode::Device, "Partitioned mode only supported with device memory");
  spdlog::info("Preparing device data.");
  auto* data = utils::map(region);
  spdlog::debug("Finished mapping memory region.");

  if (config.is_generate_shuffled_access_positions()) {
    // We use 64 B alignment as we also support clwb and clflush(opt), which require 64 B alignment.
    utils::generate_shuffled_access_positions(data, region, config, 64);
    spdlog::debug("Finished generating shuffled access positions.");
    if (!utils::verify_shuffled_access_positions(data, region, config, 64)) {
      spdlog::critical("Verifying shuffled access positions failed.");
      utils::crash_exit();
    }
  } else if (config.is_generate_read_data()) {
    // If we read data in this benchmark, we need to generate it first.
    utils::generate_read_data(data, region.size);
    spdlog::debug("Finished generating read data.");
  }

  spdlog::info("Finished preparing device data.");
  return data;
}

char* Benchmark::prepare_interleaved_data(const MemoryRegionDefinition& region, const BenchmarkConfig& config) {
  BenchAssert(region.memory_mode() == MemoryMode::Numa,
              "Partitioned mode only supported with NUMA"
              "memory");
  spdlog::info("Preparing interleaved data.");
  auto* data = utils::map(region);
  bind_memory_interleaved(data, region.size, region.node_ids);
  spdlog::debug("Finished mapping memory region.");
  utils::populate_memory(data, region.size);
  spdlog::debug("Finished populating/pre-faulting the memory region.");

  if (config.is_generate_shuffled_access_positions()) {
    utils::generate_shuffled_access_positions(data, region, config, 64);
    spdlog::debug("Finished generating shuffled access positions.");
    if (!utils::verify_shuffled_access_positions(data, region, config, 64)) {
      spdlog::critical("Verifying shuffled access positions failed.");
      utils::crash_exit();
    }
  } else if (config.is_generate_read_data()) {
    // If we read data in this benchmark, we need to generate it first.
    utils::generate_read_data(data, region.size);
    spdlog::debug("Finished generating read data.");
  }

  if (!verify_interleaved_page_placement(data, region.size, region.node_ids)) {
    auto page_locations = PageLocations{};
    fill_page_locations_round_robin(page_locations, region.size, region.node_ids);
    place_pages(data, region.size, page_locations);
    if (!verify_interleaved_page_placement(data, region.size, region.node_ids)) {
      spdlog::critical("Verification for interleaved pages failed again. Stopping here.");
      utils::crash_exit();
    }
  }

  spdlog::info("Finished preparing interleaved data.");
  return data;
}

char* Benchmark::prepare_partitioned_data(const MemoryRegionDefinition& region, const BenchmarkConfig& config) {
  BenchAssert(region.memory_mode() == MemoryMode::Numa,
              "Partitioned mode only supported with NUMA"
              "memory");
  spdlog::info("Preparing partitioned data.");
  BenchAssert(region.percentage_pages_first_partition, "Percentage of pages in first partition is not configured.");
  BenchAssert(region.node_count_first_partition, "Number of nodes belonging to the first partition not set.");
  auto* data = utils::map(region);
  spdlog::debug("Finished mapping memory region.");

  const auto region_page_count = region.size / utils::PAGE_SIZE;
  const auto first_partition_page_count =
      static_cast<u32>((*region.percentage_pages_first_partition / 100.f) * region_page_count);

  const auto first_partition_length = first_partition_page_count * utils::PAGE_SIZE;
  const auto nodes_first_partition =
      NumaNodeIDs(region.node_ids.begin(), region.node_ids.begin() + *region.node_count_first_partition);
  bind_memory_interleaved(data, first_partition_length, nodes_first_partition);
  const auto second_partition_start = data + first_partition_length;
  const auto second_partition_length = (region_page_count - first_partition_page_count) * utils::PAGE_SIZE;
  const auto nodes_second_partition =
      NumaNodeIDs(region.node_ids.begin() + *region.node_count_first_partition, region.node_ids.end());
  bind_memory_interleaved(second_partition_start, second_partition_length, nodes_second_partition);

  utils::populate_memory(data, region.size);
  spdlog::debug("Finished populating/pre-faulting the memory region.");

  if (config.is_generate_shuffled_access_positions()) {
    utils::generate_shuffled_access_positions(data, region, config, 64);
    spdlog::debug("Finished generating shuffled access positions.");
    if (!utils::verify_shuffled_access_positions(data, region, config, 64)) {
      spdlog::critical("Verifying shuffled access positions failed.");
      utils::crash_exit();
    }
  } else if (config.is_generate_read_data()) {
    // If we read data in this benchmark, we need to generate it first.
    utils::generate_read_data(data, region.size);
    spdlog::debug("Finished generating read data.");
  }

  if (!verify_partitioned_page_placement(data, region.size, region.node_ids, *region.percentage_pages_first_partition,
                                         *region.node_count_first_partition)) {
    spdlog::critical("Page verification for partitioned memory region failed.");
    utils::crash_exit();
  }

  spdlog::info("Finished preparing partitioned data.");
  return data;
}

void Benchmark::verify_page_locations(const MemoryRegions& memory_regions,
                                      const MemoryRegionDefinitions& region_definitions, const u32 workload_idx) {
  BenchAssert(memory_regions.size() == MEM_REGION_COUNT, "Number of passed memory regions incorrect.");
  BenchAssert(region_definitions.size() == MEM_REGION_COUNT, "Number of passed memory region definitions incorrect.");
  for (auto region_idx = u64{0}; region_idx < MEM_REGION_COUNT; ++region_idx) {
    spdlog::info("Verify page locations of workload {} region {}.", workload_idx, region_idx);
    auto verified = false;
    auto& region = memory_regions[region_idx];
    auto& definition = region_definitions[region_idx];
    if (definition.size == 0) {
      spdlog::info("Skipping verification for memory region with size 0.");
      continue;
    }
    if (definition.memory_mode() == MemoryMode::Device) {
      spdlog::info("Skipping verification for memory mode 'Device'.");
      continue;
    }
    if (definition.placement_mode() == PagePlacementMode::NumaInterleaved) {
      verified = verify_interleaved_page_placement(region, definition.size, definition.node_ids);
    } else {
      BenchAssert(definition.percentage_pages_first_partition,
                  "Percentage for page placement must be set for partitioned verificatiopn");
      verified = verify_partitioned_page_placement(region, definition.size, definition.node_ids,
                                                   *definition.percentage_pages_first_partition,
                                                   *definition.node_count_first_partition);
    }
    if (!verified) {
      spdlog::critical("Page locations of region {} incorrect.", region_idx);
    }
    spdlog::info("Finished verifying page locations of workload {} and memory region {}.", workload_idx, region_idx);
  }
}

void Benchmark::run_custom_ops_in_thread(ThreadConfig* thread_config, const BenchmarkConfig& config) {
  const auto& operations = config.custom_operations;
  const auto num_ops = operations.size();

  auto operation_chain = std::vector<ChainedOperation>{};
  operation_chain.reserve(num_ops);

  // Determine maximum access size to ensure that operations don't write beyond the end of the range.
  auto max_access_size = u64{0};
  for (const CustomOp& op : operations) {
    max_access_size = std::max(op.size, max_access_size);
  }

  const auto aligned_range_size = u64{thread_config->primary_region_size - max_access_size};
  const auto aligned_secondary_range_size = u64{thread_config->secondary_region_size - max_access_size};

  for (auto op_idx = u64{0}; op_idx < num_ops; ++op_idx) {
    const CustomOp& op = operations[op_idx];

    if (op.memory_type == MemoryType::Primary) {
      operation_chain.emplace_back(op, thread_config->primary_start_addr, aligned_range_size);
    } else {
      operation_chain.emplace_back(op, thread_config->secondary_start_addr, aligned_secondary_range_size);
    }

    if (op_idx > 0) {
      operation_chain[op_idx - 1].set_next(&operation_chain[op_idx]);
    }
  }

  const auto seed = std::chrono::steady_clock::now().time_since_epoch().count() * (thread_config->thread_idx + 1);
  lehmer64_seed(seed);
  char* start_addr = reinterpret_cast<char*>(seed);

  ChainedOperation& start_op = operation_chain[0];
  auto start_ts = std::chrono::steady_clock::now();

  const auto ops_count_per_batch = thread_config->ops_count_per_batch;
  auto total_num_ops = u64{0};

  while (true) {
    if (thread_config->execution->num_custom_batches_remaining.fetch_sub(1) <= 0) {
      break;
    }

    if (config.latency_sample_frequency == 0) {
      // We don't want the sampling code overhead if we don't want to sample the latency.
      for (size_t iteration = 0; iteration < ops_count_per_batch; ++iteration) {
        start_op.run(start_addr, '0');
      }
    } else {
      // Latency sampling requested, measure the latency every x iterations.
      const u64 freq = config.latency_sample_frequency;
      // Start at 1 to avoid measuring latency of first request.
      for (size_t iteration = 1; iteration <= ops_count_per_batch; ++iteration) {
        if (iteration % freq == 0) {
          auto op_start = std::chrono::steady_clock::now();
          start_op.run(start_addr, '0');
          auto op_end = std::chrono::steady_clock::now();
          thread_config->op_latencies->emplace_back((op_end - op_start).count());
        } else {
          start_op.run(start_addr, '0');
        }
      }
    }

    total_num_ops += ops_count_per_batch;
  }

  auto end_ts = std::chrono::steady_clock::now();
  *(thread_config->total_operation_duration) = ExecutionDuration{start_ts, end_ts};
  *(thread_config->total_operation_size) = total_num_ops;
}

void Benchmark::run_in_thread(ThreadConfig* thread_config, const BenchmarkConfig& config) {
  // Pin thread to the configured numa nodes.
  auto& expected_run_cores = thread_config->affinity_core_ids;
  pin_thread_to_cores(expected_run_cores);
  log_permissions_for_numa_nodes(spdlog::level::debug, thread_config->thread_idx);

  // Check if thread is pinned to the configured NUMA node.
  const auto allowed_cores = allowed_thread_core_ids();
  if (!expected_run_cores.empty() && !utils::is_subset<u64>(allowed_cores, expected_run_cores)) {
    spdlog::critical("Thread #{}: Thread not pinned to the configured CPUs. Expected: [{}], Allowed: [{}]",
                     thread_config->thread_idx, utils::numbers_to_string(expected_run_cores),
                     utils::numbers_to_string(allowed_cores));
    utils::crash_exit();
  }

  if (config.exec_mode == Mode::Custom) {
    return run_custom_ops_in_thread(thread_config, config);
  }

  if (config.is_latency_mode()) {
    return run_latency_measurements_in_thread(thread_config, config);
  }

  const size_t seed = std::chrono::steady_clock::now().time_since_epoch().count() * (thread_config->thread_idx + 1);
  lehmer64_seed(seed);

  const u32 access_count_in_range = thread_config->primary_region_size / config.access_size;

  auto access_distribution = [&]() { return lehmer64() % access_count_in_range; };

  spdlog::debug("Thread {}: Starting address generation", thread_config->thread_idx);
  const auto generation_begin_ts = std::chrono::steady_clock::now();

  // Create all batches before executing.
  size_t batch_count_per_thread = thread_config->batch_count / config.number_threads;
  const size_t remaining_batch_count = thread_config->batch_count % config.number_threads;
  if (remaining_batch_count > 0 && thread_config->thread_idx < remaining_batch_count) {
    // This thread needs to create an extra batch for an uneven number.
    ++batch_count_per_thread;
  }

  const size_t thread_idx = thread_config->thread_idx;
  const size_t thread_offset = thread_idx * config.min_io_batch_size;
  // The size in bytes that all threads read. Assuming n batches C_n and 4 partition threads PT_m, then PT_0,
  // PT_1, PT_2, PT_3 will initally access C_0, C_1, C_2, and C_4 respectively. If PT_0 finished accessing C_0, it will
  // continue with accessing with C_5. The offset delta between C_0's and C_5's start address is 4 * batch size.
  const size_t threads_access_size = thread_config->thread_count * config.min_io_batch_size;

  for (size_t batch_idx = 0; batch_idx < batch_count_per_thread; ++batch_idx) {
    const size_t thread_batch_offset =
        // Overall offset after x batches          + offset of this thread for batch x+1
        (batch_idx * threads_access_size) + thread_offset;
    char* next_op_position = config.exec_mode == Mode::Sequential_Desc
                                 ? thread_config->primary_start_addr - thread_batch_offset
                                 : thread_config->primary_start_addr + thread_batch_offset;

    std::vector<char*> op_addresses(thread_config->ops_count_per_batch);

    for (size_t op_idx = 0; op_idx < thread_config->ops_count_per_batch; ++op_idx) {
      switch (config.exec_mode) {
        case Mode::Random: {
          char* primary_start_addr;
          std::function<u64()> random_distribution;
          primary_start_addr = thread_config->primary_start_addr;
          random_distribution = access_distribution;

          u64 random_value;
          // Get a random number in the range [0, target_access_count_in_range - 1].
          if (config.random_distribution == RandomDistribution::Uniform) {
            random_value = random_distribution();
          } else {
            random_value = utils::zipf(config.zipf_alpha, access_count_in_range);
          }
          op_addresses[op_idx] = primary_start_addr + (random_value * config.access_size);
          break;
        }
        case Mode::Sequential: {
          op_addresses[op_idx] = next_op_position;
          next_op_position += config.access_size;
          break;
        }
        case Mode::Sequential_Desc: {
          op_addresses[op_idx] = next_op_position;
          next_op_position -= config.access_size;
          break;
        }
        default: {
          spdlog::critical("Illegal state. Cannot be in `run_in_thread()` with different mode.");
          utils::crash_exit();
        }
      }
    }

    // We can always pass the cache_instruction as is. It is ignored for read access.
    const size_t insert_pos = (batch_idx * config.number_threads) + thread_config->thread_idx;

    AccessBatch& current_batch = thread_config->execution->access_batches[insert_pos];
    current_batch.addresses_ = std::move(op_addresses);
    current_batch.access_size_ = config.access_size;
    current_batch.op_type_ = config.operation;
    current_batch.cache_instruction_ = config.cache_instruction;
  }

  const auto generation_end_ts = std::chrono::steady_clock::now();
  const u64 generation_duration_us =
      std::chrono::duration_cast<std::chrono::milliseconds>(generation_end_ts - generation_begin_ts).count();
  spdlog::debug("Thread {}: Finished address generation in {} ms", thread_config->thread_idx, generation_duration_us);

  auto is_last = bool{false};
  u16& threads_remaining = thread_config->execution->threads_remaining;
  {
    std::lock_guard<std::mutex> gen_lock{thread_config->execution->generation_lock};
    threads_remaining -= 1;
    is_last = threads_remaining == 0;
  }

  if (is_last) {
    thread_config->execution->generation_done.notify_all();
  } else {
    std::unique_lock<std::mutex> gen_lock{thread_config->execution->generation_lock};
    thread_config->execution->generation_done.wait(gen_lock, [&] { return threads_remaining == 0; });
  }

  // Generation is done in all threads, start execution
  const auto execution_begin_ts = std::chrono::steady_clock::now();
  std::atomic<u64>* batch_position = &thread_config->execution->batch_position;

  auto executed_op_count = u64{0};
  if (config.run_time == 0) {
    executed_op_count = run_fixed_sized_benchmark(&thread_config->execution->access_batches, batch_position);
  } else {
    const auto execution_end = execution_begin_ts + std::chrono::seconds{config.run_time};
    executed_op_count =
        run_duration_based_benchmark(&thread_config->execution->access_batches, batch_position, execution_end);
  }

  const auto execution_end_ts = std::chrono::steady_clock::now();
  const auto execution_duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(execution_end_ts - execution_begin_ts);
  spdlog::debug("Thread {}: Finished execution in {} ms", thread_config->thread_idx, execution_duration.count());

  const u64 batch_size = config.access_size * thread_config->ops_count_per_batch;
  *(thread_config->total_operation_size) = executed_op_count * batch_size;
  *(thread_config->total_operation_duration) = ExecutionDuration{execution_begin_ts, execution_end_ts};
}

u64 Benchmark::run_fixed_sized_benchmark(std::vector<AccessBatch>* access_batches, std::atomic<u64>* batch_position) {
  const u64 total_op_count = access_batches->size();
  u64 executed_op_count = 0;

  while (true) {
    const u64 batch_pos = batch_position->fetch_add(1);
    if (batch_pos >= total_op_count) {
      break;
    }

    (*access_batches)[batch_pos].run();
    ++executed_op_count;
  }

  return executed_op_count;
}

u64 Benchmark::run_duration_based_benchmark(std::vector<AccessBatch>* access_batches, std::atomic<u64>* batch_position,
                                            std::chrono::steady_clock::time_point execution_end) {
  const u64 total_op_count = access_batches->size();
  u64 executed_op_count = 0;

  while (true) {
    const u64 work_package = batch_position->fetch_add(1) % total_op_count;

    (*access_batches)[work_package].run();
    ++executed_op_count;

    const auto current_time = std::chrono::steady_clock::now();
    if (current_time > execution_end) {
      break;
    }
  }

  return executed_op_count;
}

// Latency measurement functions ---------------------------------------------------------------------------------------

void Benchmark::run_latency_measurements_in_thread(ThreadConfig* thread_config, const BenchmarkConfig& config) {
  auto* buffer = reinterpret_cast<std::byte*>(thread_config->primary_start_addr);
  const auto access_count = config.number_operations;
  spdlog::info("Access count: {}", access_count);
  thread_config->op_latencies->reserve(access_count / config.latency_sample_frequency);

  auto run_measurements = [&]() {
    if (config.is_memory_management_op()) {
      switch (config.operation) {
        case Operation::MemoryMapShared:
          return run_memory_map_shared(buffer, config, thread_config->op_latencies);
        case Operation::MemoryMapPrivate:
          return run_memory_map_private(buffer, config, thread_config->op_latencies);
        case Operation::MemoryUnmapShared:
          return run_memory_unmap_shared(buffer, config, thread_config->op_latencies);
        case Operation::MemoryUnmapPrivate:
          return run_memory_unmap_private(buffer, config, thread_config->op_latencies);
        default:
          throw BenchException("Unsupported memory management operation.");
      }
    }

    switch (config.access_size) {
      case 8:  //  ---------------------------------------------------------------------------------------------------
        if (config.exec_mode == Mode::RandomLatency && config.operation == Operation::Read &&
            config.cache_instruction == CacheInstruction::None) {
          spdlog::info("Thread {}: Running 8B random reads", thread_config->thread_idx);
          return run_random_reads_8B(buffer, access_count, config, thread_config->op_latencies);
        }
        if (config.exec_mode == Mode::RandomLatency && config.operation == Operation::Read &&
            config.cache_instruction == CacheInstruction::FlushOpt) {
          spdlog::info("Thread {}: Running 8B random reads (flush opt)", thread_config->thread_idx);
          return run_random_reads_8B_flushed(buffer, access_count, config, thread_config->op_latencies);
        }
        if (config.exec_mode == Mode::RandomLatency && config.operation == Operation::Write &&
            config.cache_instruction == CacheInstruction::WriteBack) {
          spdlog::info("Thread {}: Running 8B random writes (write back)", thread_config->thread_idx);
          return run_random_writes_8B(buffer, access_count, config, thread_config->op_latencies);
        }
        if (config.exec_mode == Mode::SequentialLatency && config.operation == Operation::Read &&
            config.cache_instruction == CacheInstruction::None) {
          spdlog::info("Thread {}: Running 8B sequential reads", thread_config->thread_idx);
          return run_sequential_reads_8B(buffer, access_count, config, thread_config->op_latencies);
        }
        if (config.exec_mode == Mode::Latency && config.operation == Operation::CompareAndSwap &&
            config.cache_instruction == CacheInstruction::None) {
          spdlog::info("Thread {}: Running 8 B compare-and-swap", thread_config->thread_idx);
          return run_atomic_cas_8B(buffer, access_count, config, thread_config->op_latencies);
        }
        if (config.exec_mode == Mode::Latency && config.operation == Operation::FetchAndAdd &&
            config.cache_instruction == CacheInstruction::None) {
          spdlog::info("Thread {}: Running 8 B fetch-and-add", thread_config->thread_idx);
          return run_atomic_faa_8B(buffer, access_count, config, thread_config->op_latencies);
        }
        throw BenchException("Mode not supported for 8 B access size.");
      case 64:  //  ---------------------------------------------------------------------------------------------------
        if (config.exec_mode == Mode::RandomLatency && config.operation == Operation::Read &&
            config.cache_instruction == CacheInstruction::None) {
          spdlog::info("Thread {}: Running 64B random reads", thread_config->thread_idx);
          return run_random_reads_64B(buffer, access_count, config, thread_config->op_latencies);
        }
        if (config.exec_mode == Mode::RandomLatency && config.operation == Operation::Read &&
            config.cache_instruction == CacheInstruction::FlushOpt) {
          spdlog::info("Thread {}: Running 64B random reads (flush opt)", thread_config->thread_idx);
          return run_random_reads_64B_flushed(buffer, access_count, config, thread_config->op_latencies);
        }
        if (config.exec_mode == Mode::RandomLatency && config.operation == Operation::StreamRead &&
            config.cache_instruction == CacheInstruction::None) {
          spdlog::info("Thread {}: Running 64B random streaming reads", thread_config->thread_idx);
          return run_random_stream_reads_64B(buffer, access_count, config, thread_config->op_latencies);
        }
        if (config.exec_mode == Mode::SequentialLatency && config.operation == Operation::Read &&
            config.cache_instruction == CacheInstruction::None) {
          spdlog::info("Thread {}: Running 64B sequential reads", thread_config->thread_idx);
          return run_sequential_reads_64B(buffer, access_count, config, thread_config->op_latencies);
        }
        if (config.exec_mode == Mode::SequentialLatency && config.operation == Operation::Read &&
            config.cache_instruction == CacheInstruction::FlushOpt) {
          spdlog::info("Thread {}: Running 64B sequential reads (flush opt)", thread_config->thread_idx);
          return run_sequential_reads_64B_flushed(buffer, access_count, config, thread_config->op_latencies);
        }
        if (config.exec_mode == Mode::SequentialLatency && config.operation == Operation::StreamRead &&
            config.cache_instruction == CacheInstruction::None) {
          spdlog::info("Thread {}: Running 64B sequential stream reads", thread_config->thread_idx);
          return run_sequential_stream_reads_64B(buffer, access_count, config, thread_config->op_latencies);
        }
        if (config.exec_mode == Mode::RandomLatency && config.operation == Operation::Write &&
            config.cache_instruction == CacheInstruction::WriteBack) {
          spdlog::info("Thread {}: Running random writes", thread_config->thread_idx);
          return run_random_writes_64B(buffer, access_count, config, thread_config->op_latencies);
        }
        if (config.exec_mode == Mode::RandomLatency && config.operation == Operation::StreamWrite &&
            config.cache_instruction == CacheInstruction::None) {
          spdlog::info("Thread {}: Running random writes", thread_config->thread_idx);
          return run_random_stream_writes_64B(buffer, access_count, config, thread_config->op_latencies);
        }
        throw BenchException("Mode not supported for 64 B access size.");
      default:
        throw BenchException("Latency measurements only supports 8 B and 64 B.");
    }
    return ExecutionDuration{};
  };

  auto exec_duration = run_measurements();

  auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(exec_duration.end - exec_duration.begin);
  spdlog::debug("Thread {}: Finished execution in {} ms", thread_config->thread_idx, duration.count());
  auto average_latency = duration.count() / access_count;
  spdlog::info("Thread {}: Average access latency: {} ns", thread_config->thread_idx, average_latency);

  const auto batch_size = config.min_io_batch_size;
  *(thread_config->total_operation_size) = access_count * config.access_size;
  *(thread_config->total_operation_duration) = exec_duration;
}

// Latency functions 8 B -----------------------------------------------------------------------------------------------

ExecutionDuration Benchmark::run_random_reads_8B(std::byte* buffer, u32 access_count, const BenchmarkConfig& config,
                                                 std::vector<u64>* latencies) {
  BenchAssert(config.access_size == 8, "Random reads support only 8 B.");
  auto sample_counter = config.latency_sample_frequency;
  auto next_pos = u64{0};
  auto begin_ts = std::chrono::steady_clock::now();
  for (auto op_idx = u64{0}; op_idx < access_count; ++op_idx) {
    // Latency measurement per operation
    // For latency measurements with random patterns, addresses are 64 B aligned to be able to execute clwb or flushes.
    auto addr = buffer + (next_pos * 64);
    if (sample_counter == 0) {
      _mm_mfence();
      // clear instruction pipeline
      rw_ops::x100_nop();
      auto op_start = std::chrono::steady_clock::now();
      next_pos = rw_ops::read_8_get_u64(reinterpret_cast<char*>(addr));
      _mm_mfence();
      auto op_end = std::chrono::steady_clock::now();
      latencies->emplace_back((op_end - op_start).count());
      sample_counter = config.latency_sample_frequency;
    } else {
      // No latency measurement per operation
      next_pos = rw_ops::read_8_get_u64(reinterpret_cast<char*>(addr));
    }
    sample_counter--;
  }
  auto end_ts = std::chrono::steady_clock::now();
  latencies->shrink_to_fit();
  return {begin_ts, end_ts};
}

ExecutionDuration Benchmark::run_random_reads_8B_flushed(std::byte* buffer, u32 access_count,
                                                         const BenchmarkConfig& config, std::vector<u64>* latencies) {
  BenchAssert(config.access_size == 8, "Random reads support only 8 B.");
  auto sample_counter = config.latency_sample_frequency;
  auto next_pos = u64{0};
  auto begin_ts = std::chrono::steady_clock::now();
  for (auto op_idx = u64{0}; op_idx < access_count; ++op_idx) {
    // Latency measurement per operation
    // For latency measurements with random patterns, addresses are 64 B aligned to be able to execute clwb or flushes.
    auto addr = buffer + (next_pos * 64u);
    if (sample_counter == 0) {
      _mm_mfence();
      // clear instruction pipeline
      rw_ops::x100_nop();
      auto op_start = std::chrono::steady_clock::now();
      _mm_clflushopt(addr);
      _mm_mfence();
      next_pos = rw_ops::read_8_get_u64(reinterpret_cast<char*>(addr));
      _mm_mfence();
      auto op_end = std::chrono::steady_clock::now();
      latencies->emplace_back((op_end - op_start).count());
      sample_counter = config.latency_sample_frequency;
    } else {
      // No latency measurement per operation
      _mm_clflushopt(addr);
      _mm_mfence();
      next_pos = rw_ops::read_8_get_u64(reinterpret_cast<char*>(addr));
    }
    sample_counter--;
  }
  auto end_ts = std::chrono::steady_clock::now();
  latencies->shrink_to_fit();
  return {begin_ts, end_ts};
}

ExecutionDuration Benchmark::run_random_writes_8B(std::byte* buffer, u32 access_count, const BenchmarkConfig& config,
                                                  std::vector<u64>* latencies) {
  BenchAssert(config.access_size == 8, "Random writes support only 8 B.");
  auto sample_counter = config.latency_sample_frequency;
  auto next_pos = u64{0};
  auto begin_ts = std::chrono::steady_clock::now();
  for (auto op_idx = u64{0}; op_idx < access_count; ++op_idx) {
    // Latency measurement per operation
    // For latency measurements with random patterns, addresses are 64 B aligned to be able to execute clwb or flushes.
    auto addr = buffer + (next_pos * 64u);
    if (sample_counter == 0) {
      _mm_mfence();
      rw_ops::x100_nop();
      next_pos = rw_ops::read_8_get_u64(reinterpret_cast<char*>(addr));
      _mm_mfence();
      auto op_start = std::chrono::steady_clock::now();
      rw_ops::write_none_8(reinterpret_cast<char*>(addr));
      _mm_clwb(addr);
      _mm_mfence();
      auto op_end = std::chrono::steady_clock::now();
      latencies->emplace_back((op_end - op_start).count());
      sample_counter = config.latency_sample_frequency;
    } else {
      // No latency measurement per operation
      next_pos = rw_ops::read_8_get_u64(reinterpret_cast<char*>(addr));
      rw_ops::write_none_8(reinterpret_cast<char*>(addr));
      _mm_clwb(addr);
      _mm_mfence();
    }
    sample_counter--;
  }
  auto end_ts = std::chrono::steady_clock::now();
  return {begin_ts, end_ts};
}

ExecutionDuration Benchmark::run_sequential_reads_8B(std::byte* buffer, u32 access_count, const BenchmarkConfig& config,
                                                     std::vector<u64>* latencies) {
  BenchAssert(config.access_size == 8, "Sequential reads support only 8 B.");
  auto sample_counter = config.latency_sample_frequency;
  auto addr_offset = u64{0};
  auto begin_ts = std::chrono::steady_clock::now();
  for (auto op_idx = u64{0}; op_idx < access_count; ++op_idx) {
    // Latency measurement per operation
    auto addr = buffer + addr_offset;
    if (sample_counter == 0) {
      _mm_mfence();
      rw_ops::x100_nop();
      auto op_start = std::chrono::steady_clock::now();
      rw_ops::read_8_get_u64(reinterpret_cast<char*>(addr));
      _mm_mfence();
      auto op_end = std::chrono::steady_clock::now();
      addr_offset += config.access_size;
      latencies->emplace_back((op_end - op_start).count());
      sample_counter = config.latency_sample_frequency;
    } else {
      // No latency measurement per operation
      rw_ops::read_8_get_u64(reinterpret_cast<char*>(addr));
      addr_offset += config.access_size;
    }
    sample_counter--;
  }
  auto end_ts = std::chrono::steady_clock::now();
  return {begin_ts, end_ts};
}

// Latency functions 64 B ----------------------------------------------------------------------------------------------

ExecutionDuration Benchmark::run_random_reads_64B(std::byte* buffer, u32 access_count, const BenchmarkConfig& config,
                                                  std::vector<u64>* latencies) {
  BenchAssert(config.access_size == 64, "Random reads support only 64 B.");
  auto sample_counter = config.latency_sample_frequency;
  auto next_pos = u64{0};
  auto begin_ts = std::chrono::steady_clock::now();
  for (auto op_idx = u64{0}; op_idx < access_count; ++op_idx) {
    // Latency measurement per operation
    // For latency measurements with random patterns, addresses are 64 B aligned to be able to execute clwb or flushes.
    auto addr = buffer + (next_pos * 64u);
    if (sample_counter == 0) {
      _mm_mfence();
      // clear instruction pipeline
      rw_ops::x100_nop();
      auto op_start = std::chrono::steady_clock::now();
      next_pos = rw_ops::read_64_get_u64(reinterpret_cast<char*>(addr));
      _mm_mfence();
      auto op_end = std::chrono::steady_clock::now();
      latencies->emplace_back((op_end - op_start).count());
      sample_counter = config.latency_sample_frequency;
    } else {
      // No latency measurement per operation
      next_pos = rw_ops::read_64_get_u64(reinterpret_cast<char*>(addr));
    }
    sample_counter--;
  }
  auto end_ts = std::chrono::steady_clock::now();
  latencies->shrink_to_fit();
  return {begin_ts, end_ts};
}

ExecutionDuration Benchmark::run_random_reads_64B_flushed(std::byte* buffer, u32 access_count,
                                                          const BenchmarkConfig& config, std::vector<u64>* latencies) {
  BenchAssert(config.access_size == 64, "Random reads support only 64 B.");
  auto sample_counter = config.latency_sample_frequency;
  auto next_pos = u64{0};
  auto begin_ts = std::chrono::steady_clock::now();
  for (auto op_idx = u64{0}; op_idx < access_count; ++op_idx) {
    // Latency measurement per operation
    auto addr = buffer + (next_pos * 64u);
    if (sample_counter == 0) {
      _mm_mfence();
      // clear instruction pipeline
      rw_ops::x100_nop();
      auto op_start = std::chrono::steady_clock::now();
      _mm_clflushopt(addr);
      _mm_mfence();
      next_pos = rw_ops::read_64_get_u64(reinterpret_cast<char*>(addr));
      _mm_mfence();
      auto op_end = std::chrono::steady_clock::now();
      latencies->emplace_back((op_end - op_start).count());
      sample_counter = config.latency_sample_frequency;
    } else {
      // No latency measurement per operation
      _mm_clflushopt(addr);
      _mm_mfence();
      next_pos = rw_ops::read_64_get_u64(reinterpret_cast<char*>(addr));
    }
    sample_counter--;
  }
  auto end_ts = std::chrono::steady_clock::now();
  latencies->shrink_to_fit();
  return {begin_ts, end_ts};
}

ExecutionDuration Benchmark::run_random_stream_reads_64B(std::byte* buffer, u32 access_count,
                                                         const BenchmarkConfig& config, std::vector<u64>* latencies) {
  BenchAssert(config.access_size == 64, "Random reads support only 64 B.");
  auto sample_counter = config.latency_sample_frequency;
  auto next_pos = u64{0};
  auto begin_ts = std::chrono::steady_clock::now();
  for (auto op_idx = u64{0}; op_idx < access_count; ++op_idx) {
    // Latency measurement per operation
    // For latency measurements with random patterns, addresses are 64 B aligned to be able to execute clwb or flushes.
    auto addr = buffer + (next_pos * 64u);
    if (sample_counter == 0) {
      _mm_mfence();
      // clear instruction pipeline
      rw_ops::x100_nop();
      auto op_start = std::chrono::steady_clock::now();
      next_pos = rw_ops::read_64_stream_get_u64(reinterpret_cast<char*>(addr));
      _mm_mfence();
      auto op_end = std::chrono::steady_clock::now();
      latencies->emplace_back((op_end - op_start).count());
      sample_counter = config.latency_sample_frequency;
    } else {
      // No latency measurement per operation
      next_pos = rw_ops::read_64_stream_get_u64(reinterpret_cast<char*>(addr));
    }
    sample_counter--;
  }
  auto end_ts = std::chrono::steady_clock::now();
  latencies->shrink_to_fit();
  return {begin_ts, end_ts};
}

ExecutionDuration Benchmark::run_random_writes_64B(std::byte* buffer, u32 access_count, const BenchmarkConfig& config,
                                                   std::vector<u64>* latencies) {
  BenchAssert(config.access_size == 64, "Random writes support only 64 B.");
  auto sample_counter = config.latency_sample_frequency;
  auto next_pos = u64{0};
  auto begin_ts = std::chrono::steady_clock::now();
  for (auto op_idx = u64{0}; op_idx < access_count; ++op_idx) {
    // Latency measurement per operation
    // For latency measurements with random patterns, addresses are 64 B aligned to be able to execute clwb or flushes.
    auto addr = buffer + (next_pos * 64u);
    if (sample_counter == 0) {
      _mm_mfence();
      rw_ops::x100_nop();
      next_pos = rw_ops::read_8_get_u64(reinterpret_cast<char*>(addr));
      _mm_mfence();
      auto op_start = std::chrono::steady_clock::now();
      rw_ops::write_none_64(reinterpret_cast<char*>(addr));
      _mm_clwb(addr);
      _mm_mfence();
      auto op_end = std::chrono::steady_clock::now();
      latencies->emplace_back((op_end - op_start).count());
      sample_counter = config.latency_sample_frequency;
    } else {
      // No latency measurement per operation
      next_pos = rw_ops::read_8_get_u64(reinterpret_cast<char*>(addr));
      rw_ops::write_none_64(reinterpret_cast<char*>(addr));
      _mm_mfence();
    }
    sample_counter--;
  }
  auto end_ts = std::chrono::steady_clock::now();
  return {begin_ts, end_ts};
}

ExecutionDuration Benchmark::run_random_stream_writes_64B(std::byte* buffer, u32 access_count,
                                                          const BenchmarkConfig& config, std::vector<u64>* latencies) {
  BenchAssert(config.access_size == 64, "Random writes support only 64 B.");
  auto sample_counter = config.latency_sample_frequency;
  auto next_pos = u64{0};
  auto begin_ts = std::chrono::steady_clock::now();
  for (auto op_idx = u64{0}; op_idx < access_count; ++op_idx) {
    // Latency measurement per operation
    // For latency measurements with random patterns, addresses are 64 B aligned to be able to execute clwb or flushes.
    auto addr = buffer + (next_pos * 64u);
    if (sample_counter == 0) {
      _mm_mfence();
      rw_ops::x100_nop();
      next_pos = rw_ops::read_8_get_u64(reinterpret_cast<char*>(addr));
      _mm_mfence();
      auto op_start = std::chrono::steady_clock::now();
      rw_ops::write_stream_64B(reinterpret_cast<char*>(addr));
      _mm_mfence();
      auto op_end = std::chrono::steady_clock::now();
      latencies->emplace_back((op_end - op_start).count());
      sample_counter = config.latency_sample_frequency;
    } else {
      // No latency measurement per operation
      next_pos = rw_ops::read_8_get_u64(reinterpret_cast<char*>(addr));
      rw_ops::write_stream_64B(reinterpret_cast<char*>(addr));
      _mm_mfence();
    }
    sample_counter--;
  }
  auto end_ts = std::chrono::steady_clock::now();
  return {begin_ts, end_ts};
}

ExecutionDuration Benchmark::run_sequential_reads_64B(std::byte* buffer, u32 access_count,
                                                      const BenchmarkConfig& config, std::vector<u64>* latencies) {
  BenchAssert(config.access_size == 64, "Sequential reads support only 64 B.");
  auto sample_counter = config.latency_sample_frequency;
  auto addr_offset = u64{0};
  auto begin_ts = std::chrono::steady_clock::now();
  for (auto op_idx = u64{0}; op_idx < access_count; ++op_idx) {
    // Latency measurement per operation
    auto addr = buffer + addr_offset;
    if (sample_counter == 0) {
      _mm_mfence();
      rw_ops::x100_nop();
      auto op_start = std::chrono::steady_clock::now();
      rw_ops::read_64_get_u64(reinterpret_cast<char*>(addr));
      _mm_mfence();
      auto op_end = std::chrono::steady_clock::now();
      addr_offset += config.access_size;
      latencies->emplace_back((op_end - op_start).count());
      sample_counter = config.latency_sample_frequency;
    } else {
      // No latency measurement per operation
      rw_ops::read_64_get_u64(reinterpret_cast<char*>(addr));
      addr_offset += config.access_size;
    }
    sample_counter--;
  }
  auto end_ts = std::chrono::steady_clock::now();
  return {begin_ts, end_ts};
}

ExecutionDuration Benchmark::run_sequential_stream_reads_64B(std::byte* buffer, u32 access_count,
                                                             const BenchmarkConfig& config,
                                                             std::vector<u64>* latencies) {
  BenchAssert(config.access_size == 64, "Sequential reads support only 64 B.");
  auto sample_counter = config.latency_sample_frequency;
  auto addr_offset = u64{0};
  auto begin_ts = std::chrono::steady_clock::now();
  for (auto op_idx = u64{0}; op_idx < access_count; ++op_idx) {
    // Latency measurement per operation
    auto addr = buffer + addr_offset;
    if (sample_counter == 0) {
      _mm_mfence();
      rw_ops::x100_nop();
      auto op_start = std::chrono::steady_clock::now();
      rw_ops::read_64_stream_get_u64(reinterpret_cast<char*>(addr));
      _mm_mfence();
      auto op_end = std::chrono::steady_clock::now();
      addr_offset += config.access_size;
      latencies->emplace_back((op_end - op_start).count());
      sample_counter = config.latency_sample_frequency;
    } else {
      // No latency measurement per operation
      rw_ops::read_64_stream_get_u64(reinterpret_cast<char*>(addr));
      addr_offset += config.access_size;
    }
    sample_counter--;
  }
  auto end_ts = std::chrono::steady_clock::now();
  return {begin_ts, end_ts};
}

ExecutionDuration Benchmark::run_sequential_reads_64B_flushed(std::byte* buffer, u32 access_count,
                                                              const BenchmarkConfig& config,
                                                              std::vector<u64>* latencies) {
  BenchAssert(config.access_size == 64, "Sequential reads support only 64 B.");
  auto sample_counter = config.latency_sample_frequency;
  auto addr_offset = u64{0};
  auto begin_ts = std::chrono::steady_clock::now();
  for (auto op_idx = u64{0}; op_idx < access_count; ++op_idx) {
    // Latency measurement per operation
    auto addr = buffer + addr_offset;
    if (sample_counter == 0) {
      _mm_mfence();
      rw_ops::x100_nop();
      auto op_start = std::chrono::steady_clock::now();
      _mm_clflushopt(addr);
      _mm_mfence();
      rw_ops::read_64_get_u64(reinterpret_cast<char*>(addr));
      _mm_mfence();
      auto op_end = std::chrono::steady_clock::now();
      addr_offset += config.access_size;
      latencies->emplace_back((op_end - op_start).count());
      sample_counter = config.latency_sample_frequency;
    } else {
      // No latency measurement per operation
      _mm_clflushopt(addr);
      _mm_mfence();
      rw_ops::read_64_get_u64(reinterpret_cast<char*>(addr));
      addr_offset += config.access_size;
    }
    sample_counter--;
  }
  auto end_ts = std::chrono::steady_clock::now();
  return {begin_ts, end_ts};
}

ExecutionDuration Benchmark::run_atomic_cas_8B(std::byte* buffer, u32 access_count, const BenchmarkConfig& config,
                                               std::vector<u64>* latencies) {
  BenchAssert(config.access_size == 8, "compare-and-swap 8B only supported for access size 8B");
  auto sample_counter = config.latency_sample_frequency;
  auto next_pos = u64{0};
  auto begin_ts = std::chrono::steady_clock::now();
  constexpr auto write_value = std::numeric_limits<u64>::max();
  for (auto op_idx = u64{0}; op_idx < access_count; ++op_idx) {
    // Latency measurement per operation
    // For latency measurements with random patterns, addresses are 64 B aligned.
    auto addr = buffer + (next_pos * 64u);
    auto* value_ptr = reinterpret_cast<u64*>(addr);
    auto atomic_value_ref = std::atomic_ref(*value_ptr);
    if (sample_counter == 0) {
      _mm_mfence();
      rw_ops::x100_nop();
      _mm_mfence();
      auto op_start = std::chrono::steady_clock::now();
      atomic_value_ref.compare_exchange_weak(next_pos, write_value, std::memory_order_seq_cst);
      auto op_end = std::chrono::steady_clock::now();
      latencies->emplace_back((op_end - op_start).count());
      sample_counter = config.latency_sample_frequency;
    } else {
      // No latency measurement per operation
      atomic_value_ref.compare_exchange_weak(next_pos, write_value, std::memory_order_seq_cst);
    }
    sample_counter--;
  }
  auto end_ts = std::chrono::steady_clock::now();
  return {begin_ts, end_ts};
}

ExecutionDuration Benchmark::run_atomic_faa_8B(std::byte* buffer, u32 access_count, const BenchmarkConfig& config,
                                               std::vector<u64>* latencies) {
  BenchAssert(config.access_size == 8, "fetch-and-add 8B only supported for access size 8B");
  auto sample_counter = config.latency_sample_frequency;
  auto next_pos = u64{0};
  auto begin_ts = std::chrono::steady_clock::now();
  for (auto op_idx = u64{0}; op_idx < access_count; ++op_idx) {
    // Latency measurement per operation
    // For latency measurements with random patterns, addresses are 64 B aligned.
    auto addr = buffer + (next_pos * 64u);
    auto* value_ptr = reinterpret_cast<u64*>(addr);
    auto atomic_value_ref = std::atomic_ref(*value_ptr);
    if (sample_counter == 0) {
      _mm_mfence();
      rw_ops::x100_nop();
      _mm_mfence();
      auto op_start = std::chrono::steady_clock::now();
      next_pos = atomic_value_ref.fetch_add(1, std::memory_order_seq_cst);
      auto op_end = std::chrono::steady_clock::now();
      latencies->emplace_back((op_end - op_start).count());
      sample_counter = config.latency_sample_frequency;
    } else {
      // No latency measurement per operation
      next_pos = atomic_value_ref.fetch_add(1, std::memory_order_seq_cst);
    }
    sample_counter--;
  }
  auto end_ts = std::chrono::steady_clock::now();
  return {begin_ts, end_ts};
}

ExecutionDuration Benchmark::run_memory_map_shared(std::byte* buffer, const BenchmarkConfig& config,
                                                   std::vector<u64>* latencies) {
  auto sample_counter = config.latency_sample_frequency;

  const auto& region = config.memory_regions[0];
  const auto is_device = region.memory_mode() == MemoryMode::Device;
  const auto file_descriptor = is_device ? open_device(region.device_path) : 0;

  auto do_alloc = [&] {
    if (is_device) {
      return map_shared_populate(region.size, file_descriptor, region.offset);
    } else {
      return map_shared_anonymous_populate(region.size, file_descriptor, region.offset);
    }
  };

  auto begin_ts = std::chrono::steady_clock::now();
  for (auto op_idx = u64{0}; op_idx < config.number_operations; ++op_idx) {
    if (sample_counter == 0) {
      _mm_mfence();
      rw_ops::x100_nop();
      _mm_mfence();
      auto op_start = std::chrono::steady_clock::now();
      buffer = do_alloc();
      auto op_end = std::chrono::steady_clock::now();
      latencies->emplace_back((op_end - op_start).count());
      sample_counter = config.latency_sample_frequency;
    } else {
      buffer = do_alloc();
    }
    sample_counter--;
    unmap(buffer, region.size);
  }
  auto end_ts = std::chrono::steady_clock::now();
  return {begin_ts, end_ts};
}

ExecutionDuration Benchmark::run_memory_map_private(std::byte* buffer, const BenchmarkConfig& config,
                                                    std::vector<u64>* latencies) {
  auto sample_counter = config.latency_sample_frequency;

  const auto& region = config.memory_regions[0];
  const auto is_device = region.memory_mode() == MemoryMode::Device;
  const auto file_descriptor = is_device ? open_device(region.device_path) : 0;

  auto do_alloc = [&] {
    if (is_device) {
      return map_private_populate(region.size, file_descriptor, region.offset);
    } else {
      return map_private_anonymous_populate(region.size, file_descriptor, region.offset);
    }
  };

  auto begin_ts = std::chrono::steady_clock::now();
  for (auto op_idx = u64{0}; op_idx < config.number_operations; ++op_idx) {
    if (sample_counter == 0) {
      _mm_mfence();
      rw_ops::x100_nop();
      _mm_mfence();
      auto op_start = std::chrono::steady_clock::now();
      buffer = do_alloc();
      auto op_end = std::chrono::steady_clock::now();
      latencies->emplace_back((op_end - op_start).count());
      sample_counter = config.latency_sample_frequency;
    } else {
      buffer = do_alloc();
    }
    sample_counter--;
    unmap(buffer, region.size);
  }
  auto end_ts = std::chrono::steady_clock::now();
  return {begin_ts, end_ts};
}

// TODO(MW) remote return values for some latency measurements.
ExecutionDuration Benchmark::run_memory_unmap_shared(std::byte* buffer, const BenchmarkConfig& config,
                                                     std::vector<u64>* latencies) {
  auto sample_counter = config.latency_sample_frequency;

  const auto& region = config.memory_regions[0];
  const auto is_device = region.memory_mode() == MemoryMode::Device;
  const auto file_descriptor = is_device ? open_device(region.device_path) : 0;

  auto do_alloc = [&] {
    if (is_device) {
      return map_shared_populate(region.size, file_descriptor, region.offset);
    } else {
      return map_shared_anonymous_populate(region.size, file_descriptor, region.offset);
    }
  };

  auto begin_ts = std::chrono::steady_clock::now();
  for (auto op_idx = u64{0}; op_idx < config.number_operations; ++op_idx) {
    buffer = do_alloc();
    if (sample_counter == 0) {
      _mm_mfence();
      rw_ops::x100_nop();
      _mm_mfence();
      auto op_start = std::chrono::steady_clock::now();
      unmap(buffer, region.size);
      auto op_end = std::chrono::steady_clock::now();
      latencies->emplace_back((op_end - op_start).count());
      sample_counter = config.latency_sample_frequency;
    } else {
      unmap(buffer, region.size);
    }
    sample_counter--;
  }
  auto end_ts = std::chrono::steady_clock::now();
  return {begin_ts, end_ts};
}

ExecutionDuration Benchmark::run_memory_unmap_private(std::byte* buffer, const BenchmarkConfig& config,
                                                      std::vector<u64>* latencies) {
  auto sample_counter = config.latency_sample_frequency;

  const auto& region = config.memory_regions[0];
  const auto is_device = region.memory_mode() == MemoryMode::Device;
  const auto file_descriptor = is_device ? open_device(region.device_path) : 0;

  auto do_alloc = [&] {
    if (is_device) {
      return map_private_populate(region.size, file_descriptor, region.offset);
    } else {
      return map_private_anonymous_populate(region.size, file_descriptor, region.offset);
    }
  };

  auto begin_ts = std::chrono::steady_clock::now();
  for (auto op_idx = u64{0}; op_idx < config.number_operations; ++op_idx) {
    buffer = do_alloc();
    if (sample_counter == 0) {
      _mm_mfence();
      rw_ops::x100_nop();
      _mm_mfence();
      auto op_start = std::chrono::steady_clock::now();
      unmap(buffer, region.size);
      auto op_end = std::chrono::steady_clock::now();
      latencies->emplace_back((op_end - op_start).count());
      sample_counter = config.latency_sample_frequency;
    } else {
      unmap(buffer, region.size);
    }
    sample_counter--;
  }
  auto end_ts = std::chrono::steady_clock::now();
  return {begin_ts, end_ts};
}

// ---------------------------------------------------------------------------------------------------------------------

const std::vector<BenchmarkConfig>& Benchmark::get_benchmark_configs() const { return configs_; }

const std::vector<MemoryRegions>& Benchmark::get_memory_regions() const { return memory_region_sets_; }

const std::vector<std::vector<ThreadConfig>>& Benchmark::get_thread_configs() const { return thread_configs_; }
const std::vector<std::unique_ptr<BenchmarkResult>>& Benchmark::get_benchmark_results() const { return results_; }

nlohmann::json Benchmark::get_json_config(u8 config_index) { return configs_[config_index].as_json(); }

void Benchmark::tear_down(bool force) {
  executions_.clear();
  results_.clear();

  for (auto workload_idx = u64{0}; workload_idx < memory_region_sets_.size(); ++workload_idx) {
    auto& workload_regions = memory_region_sets_[workload_idx];
    auto& region_definitions = configs_[workload_idx].memory_regions;
    BenchAssert(workload_regions.size() == MEM_REGION_COUNT, "Unexpected memory region count.");
    for (auto region_idx = u64{0}; region_idx < MEM_REGION_COUNT; ++region_idx) {
      if (workload_regions[region_idx] != nullptr) {
        munmap(workload_regions[region_idx], region_definitions[region_idx].size);
        workload_regions[region_idx] = nullptr;
      }
    }
  }
}

const std::unordered_map<std::string, BenchmarkType> BenchmarkEnums::str_to_benchmark_type{
    {"single", BenchmarkType::Single}, {"parallel", BenchmarkType::Parallel}};

BenchmarkResult::BenchmarkResult(BenchmarkConfig config) : config{std::move(config)}, latency_hdr{nullptr} {
  // Initialize HdrHistrogram
  // 100 seconds in nanoseconds as max value.
  hdr_init(1, 100000000000, 3, &latency_hdr);
}

BenchmarkResult::~BenchmarkResult() {
  if (latency_hdr != nullptr) {
    hdr_close(latency_hdr);
  }

  operation_latencies.clear();
  operation_latencies.shrink_to_fit();
}

nlohmann::json BenchmarkResult::get_result_as_json() const {
  if (config.exec_mode == Mode::Custom) {
    return get_custom_results_as_json();
  }

  if (config.is_latency_mode()) {
    return get_latency_mode_results_as_json();
  }

  if (total_operation_durations.size() != config.number_threads) {
    spdlog::critical("Invalid state! Need n result durations for n threads. Got: {} but expected: {}",
                     total_operation_durations.size(), config.number_threads);
    utils::crash_exit();
  }

  if (total_operation_sizes.size() != config.number_threads) {
    spdlog::critical("Invalid state! Need n result sizes for n threads. Got: {} but expected: {}",
                     total_operation_sizes.size(), config.number_threads);
    utils::crash_exit();
  }

  std::chrono::steady_clock::time_point earliest_begin = total_operation_durations[0].begin;
  std::chrono::steady_clock::time_point latest_end = total_operation_durations[0].end;

  auto total_size = u64{0};
  std::vector<double> per_thread_bandwidth(config.number_threads);
  std::vector<u64> per_thread_op_latency(config.number_threads);
  nlohmann::json per_thread_results = nlohmann::json::array();

  for (u16 thread_idx = 0; thread_idx < config.number_threads; ++thread_idx) {
    const ExecutionDuration& thread_timestamps = total_operation_durations[thread_idx];
    const std::chrono::steady_clock::duration thread_duration = thread_timestamps.duration();
    const auto thread_duration_s = std::chrono::duration<double>(std::chrono::nanoseconds{thread_duration}).count();

    const u64 thread_op_size = total_operation_sizes[thread_idx];

    const double thread_bandwidth = get_bandwidth(thread_op_size, thread_duration);
    per_thread_bandwidth[thread_idx] = thread_bandwidth;
    const auto access_count = thread_op_size / config.access_size;
    per_thread_op_latency[thread_idx] = std::chrono::nanoseconds{thread_duration}.count() / access_count;

    spdlog::debug("Thread {}: Per-Thread Information", thread_idx);
    spdlog::debug(" ├─ Bandwidth (GiB/s): {:.5f}", thread_bandwidth);
    spdlog::debug(" ├─ Total Access Size (MiB): {}", thread_op_size / MiB);
    spdlog::debug(" └─ Duration (s): {:.5f}", thread_duration_s);

    total_size += thread_op_size;
    earliest_begin = std::min(earliest_begin, thread_timestamps.begin);
    latest_end = std::max(latest_end, thread_timestamps.end);

    nlohmann::json thread_results;
    thread_results["bandwidth"] = thread_bandwidth;
    thread_results["execution_time"] = thread_duration_s;
    thread_results["accessed_bytes"] = thread_op_size;
    per_thread_results.emplace_back(std::move(thread_results));
  }

  const auto execution_time = latest_end - earliest_begin;
  const double total_bandwidth = get_bandwidth(total_size, execution_time);

  // Bandwidth: Add information about per-thread avg and standard deviation.
  const double avg_bandwidth = total_bandwidth / config.number_threads;
  double bandwidth_stddev = calculate_standard_deviation(per_thread_bandwidth, avg_bandwidth);

  spdlog::debug("Per-Thread Average Bandwidth: {}", avg_bandwidth);
  spdlog::debug("Per-Thread Bandwidth Standard Deviation: {}", bandwidth_stddev);
  spdlog::debug("Total Bandwidth: {}", total_bandwidth);

  // Access Latency: Add information about per-thread avg and standard deviation.
  const double total_threads_op_latency =
      std::accumulate(per_thread_op_latency.begin(), per_thread_op_latency.end(), 0);
  const double avg_op_latency = (total_threads_op_latency / config.number_threads);
  double op_latency_stddev = calculate_standard_deviation(per_thread_op_latency, avg_op_latency);

  spdlog::debug("Per-Thread Average Access Latency: {}", avg_op_latency);
  spdlog::debug("Per-Thread Access Latency Standard Deviation: {}", op_latency_stddev);

  nlohmann::json results;
  nlohmann::json results_entries;

  auto execution_time_s = std::chrono::duration<double>(std::chrono::nanoseconds{execution_time});

  results_entries["start_time"] = start_timestamp.time_since_epoch().count();
  results_entries["bandwidth"] = total_bandwidth;
  results_entries["execution_time"] = execution_time_s.count();
  results_entries["accessed_bytes"] = total_size;
  results_entries["thread_bandwidth_avg"] = avg_bandwidth;
  results_entries["thread_bandwidth_std_dev"] = bandwidth_stddev;
  results_entries["thread_op_latency_avg"] = avg_op_latency;
  results_entries["thread_op_latency_std_dev"] = op_latency_stddev;
  results_entries["threads"] = per_thread_results;

  results["results"] = results_entries;

  if (execution_time < std::chrono::seconds{1}) {
    spdlog::warn(
        "Benchmark ran less then 1 second ({} ms). The results may be inaccurate due to the short execution time.",
        std::chrono::duration_cast<std::chrono::milliseconds>(execution_time).count());
  }

  return results;
}

nlohmann::json BenchmarkResult::get_custom_results_as_json() const {
  nlohmann::json custom_op_results;
  nlohmann::json per_thread_results = nlohmann::json::array();

  u64 total_num_ops = 0;
  std::vector<double> per_thread_ops_per_s(config.number_threads);

  std::chrono::steady_clock::time_point earliest_begin = total_operation_durations[0].begin;
  std::chrono::steady_clock::time_point latest_end = total_operation_durations[0].end;

  for (u64 thread_idx = 0; thread_idx < config.number_threads; ++thread_idx) {
    const ExecutionDuration& thread_timestamps = total_operation_durations[thread_idx];
    const std::chrono::steady_clock::duration thread_duration = thread_timestamps.duration();
    const auto thread_duration_s = std::chrono::duration<double>(std::chrono::nanoseconds{thread_duration}).count();

    const u64 num_ops = total_operation_sizes[thread_idx];
    total_num_ops += num_ops;
    const double thread_ops_per_s = static_cast<double>(num_ops) / thread_duration_s;
    per_thread_ops_per_s[thread_idx] = thread_ops_per_s;

    earliest_begin = std::min(earliest_begin, thread_timestamps.begin);
    latest_end = std::max(latest_end, thread_timestamps.end);

    nlohmann::json thread_results;
    thread_results["num_operations"] = num_ops;
    thread_results["execution_time"] = thread_duration_s;
    thread_results["ops_per_second"] = thread_ops_per_s;
    per_thread_results.emplace_back(std::move(thread_results));
  }

  const auto execution_time = latest_end - earliest_begin;
  auto execution_time_s = std::chrono::duration<double>(std::chrono::nanoseconds{execution_time});

  const double total_ops_per_s = static_cast<double>(total_num_ops) / execution_time_s.count();
  const double avg_ops_per_s = total_ops_per_s / config.number_threads;
  const double ops_per_s_std_dev = calculate_standard_deviation(per_thread_ops_per_s, avg_ops_per_s);

  custom_op_results["execution_time"] = execution_time_s.count();
  custom_op_results["num_operations"] = total_num_ops;
  custom_op_results["ops_per_second"] = total_ops_per_s;
  custom_op_results["thread_ops_per_second_avg"] = avg_ops_per_s;
  custom_op_results["thread_ops_per_second_std_dev"] = ops_per_s_std_dev;
  custom_op_results["threads"] = per_thread_results;

  if (config.latency_sample_frequency > 0) {
    for (const std::vector<u64>& thread_latencies : operation_latencies) {
      for (const u64 latency : thread_latencies) {
        hdr_record_value(latency_hdr, static_cast<i64>(latency));
      }
    }
    custom_op_results["latency"] = hdr_histogram_to_json(latency_hdr);
  }

  nlohmann::json result;
  result["results"] = custom_op_results;
  return result;
}

nlohmann::json BenchmarkResult::get_latency_mode_results_as_json() const {
  nlohmann::json latency_results;
  nlohmann::json per_thread_results = nlohmann::json::array();

  u64 total_num_ops = 0;
  std::vector<double> per_thread_ops_per_s(config.number_threads);

  std::chrono::steady_clock::time_point earliest_begin = total_operation_durations[0].begin;
  std::chrono::steady_clock::time_point latest_end = total_operation_durations[0].end;

  for (u64 thread_idx = 0; thread_idx < config.number_threads; ++thread_idx) {
    const ExecutionDuration& thread_timestamps = total_operation_durations[thread_idx];
    const std::chrono::steady_clock::duration thread_duration = thread_timestamps.duration();
    const auto thread_duration_s = std::chrono::duration<double>(std::chrono::nanoseconds{thread_duration}).count();

    const u64 num_ops = total_operation_sizes[thread_idx];
    total_num_ops += num_ops;
    const double thread_ops_per_s = static_cast<double>(num_ops) / thread_duration_s;
    per_thread_ops_per_s[thread_idx] = thread_ops_per_s;

    earliest_begin = std::min(earliest_begin, thread_timestamps.begin);
    latest_end = std::max(latest_end, thread_timestamps.end);

    nlohmann::json thread_results;
    thread_results["num_operations"] = num_ops;
    thread_results["execution_time"] = thread_duration_s;
    thread_results["ops_per_second"] = thread_ops_per_s;
    per_thread_results.emplace_back(std::move(thread_results));
  }

  const auto execution_time = latest_end - earliest_begin;
  auto execution_time_s = std::chrono::duration<double>(std::chrono::nanoseconds{execution_time});

  const double total_ops_per_s = static_cast<double>(total_num_ops) / execution_time_s.count();
  const double avg_ops_per_s = total_ops_per_s / config.number_threads;
  const double ops_per_s_std_dev = calculate_standard_deviation(per_thread_ops_per_s, avg_ops_per_s);

  latency_results["execution_time"] = execution_time_s.count();
  latency_results["num_operations"] = total_num_ops;
  latency_results["ops_per_second"] = total_ops_per_s;
  latency_results["thread_ops_per_second_avg"] = avg_ops_per_s;
  latency_results["thread_ops_per_second_std_dev"] = ops_per_s_std_dev;
  latency_results["threads"] = per_thread_results;
  latency_results["latencies"] = operation_latencies;

  nlohmann::json result;
  result["results"] = latency_results;
  return result;
}

}  // namespace cxlbench
