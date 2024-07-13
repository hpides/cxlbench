#include "benchmark.hpp"

#include <spdlog/spdlog.h>

#include <condition_variable>
#include <thread>
#include <utility>

#include "benchmark_config.hpp"
#include "fast_random.hpp"
#include "numa.hpp"
#include "threads.hpp"

namespace {

template <typename T>
double calculate_standard_deviation(const std::vector<T>& values, const double average) {
  const uint16_t num_values = values.size();
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

inline double get_bandwidth(const uint64_t total_data_size, const std::chrono::steady_clock::duration total_duration) {
  const double duration_in_s = static_cast<double>(total_duration.count()) / mema::SECONDS_IN_NANOSECONDS;
  const double data_in_gib = static_cast<double>(total_data_size) / mema::GiB;
  return data_in_gib / duration_in_s;
}

}  // namespace

namespace mema {

const std::string& Benchmark::benchmark_name() const { return benchmark_name_; }

void Benchmark::log_config() {}
void Benchmark::log_information() {}
void Benchmark::debug_log_json_config(size_t benchmark_idx) {}
std::string Benchmark::benchmark_type_as_str() const {
  return utils::get_enum_as_string(BenchmarkEnums::str_to_benchmark_type, benchmark_type_);
}

BenchmarkType Benchmark::get_benchmark_type() const { return benchmark_type_; }

void Benchmark::single_set_up(const BenchmarkConfig& config, MemoryRegions& memory_regions,
                              BenchmarkExecution* execution, BenchmarkResult* result,
                              std::vector<std::thread>* thread_pool, std::vector<ThreadConfig>* thread_configs) {
  const size_t total_range_op_count = config.memory_regions[0].size / config.access_size;
  const bool is_custom_execution = config.exec_mode == Mode::Custom;
  const size_t num_operations =
      (config.exec_mode == Mode::Random || is_custom_execution) ? config.number_operations : total_range_op_count;
  const size_t op_count_per_thread = num_operations / config.number_threads;

  thread_pool->reserve(config.number_threads);
  thread_configs->reserve(config.number_threads);
  result->total_operation_durations.resize(config.number_threads);
  result->total_operation_sizes.resize(config.number_threads, 0);

  uint64_t latency_measurement_count = 0;
  if (is_custom_execution) {
    result->custom_operation_latencies.resize(config.number_threads);

    if (config.latency_sample_frequency > 0) {
      latency_measurement_count = (op_count_per_thread / config.latency_sample_frequency) * 2;
    }
  }

  // If number_partitions is 0, each thread gets its own partition.
  const uint16_t partition_count = config.number_partitions == 0 ? config.number_threads : config.number_partitions;

  const uint16_t thread_count_per_partition = config.number_threads / partition_count;
  const uint64_t partition_size = config.memory_regions[0].size / partition_count;
  MemaAssert(config.memory_regions[1].size == 0 || partition_count == 1,
             "With a secondary partition size, only one partition is supported.");
  const auto secondary_partition_size = config.memory_regions[1].size;

  // Set up thread synchronization and execution parameters
  const auto& access_size =
      is_custom_execution ? CustomOp::cumulative_size(config.custom_operations) : config.access_size;
  const uint64_t ops_per_chunk = access_size < config.min_io_chunk_size ? config.min_io_chunk_size / access_size : 1;

  // Add one chunk for random execution and non-divisible numbers so that we perform at least number_operations ops and
  // not fewer. Adding a chunk in sequential access exceeds the memory range and segfaults.
  const bool is_sequential = config.exec_mode == Mode::Sequential || config.exec_mode == Mode::Sequential_Desc;
  const size_t extra_chunk = is_sequential ? 0 : (num_operations % ops_per_chunk != 0);
  const size_t chunk_count = (num_operations / ops_per_chunk) + extra_chunk;

  execution->threads_remaining = config.number_threads;
  execution->io_position = 0;
  execution->io_operations.resize(chunk_count);
  execution->num_custom_chunks_remaining = static_cast<int64_t>(chunk_count);

  auto* primary_region = memory_regions[0];
  // Assumption: only one partition exists for secondary region. Secondary region is only used for custom operations.
  MemaAssert(config.memory_regions[1].size == 0 || config.exec_mode == Mode::Custom,
             "Secondary memory region is only supported with custom operations.");
  auto* secondary_partition_start = memory_regions[1];

  // Thread pinning preparation
  const auto is_numa_thread_pinning = (config.thread_pin_mode == ThreadPinMode::AllNumaCores ||
                                       config.thread_pin_mode == ThreadPinMode::SingleNumaCoreIncrement);
  const auto threads_pinning_cores =
      is_numa_thread_pinning ? core_ids_of_nodes(config.numa_thread_nodes) : config.thread_core_ids;

  // TODO(MW) remove partitioning. Always assume one partition.
  for (uint16_t partition_idx = 0; partition_idx < partition_count; ++partition_idx) {
    char* partition_start = (config.exec_mode == Mode::Sequential_Desc)
                                ? primary_region + ((partition_count - partition_idx) * partition_size) - access_size
                                : primary_region + (partition_idx * partition_size);

    for (uint16_t partition_thread_idx = 0; partition_thread_idx < thread_count_per_partition; ++partition_thread_idx) {
      const auto thread_idx = (partition_idx * thread_count_per_partition) + partition_thread_idx;

      // Reserve space for custom operation latency measurements to avoid resizing during benchmark execution.
      if (is_custom_execution) {
        result->custom_operation_latencies[thread_idx].reserve(latency_measurement_count);
      }

      ExecutionDuration* total_op_duration = &result->total_operation_durations[thread_idx];
      uint64_t* total_op_size = &result->total_operation_sizes[thread_idx];
      std::vector<uint64_t>* custom_op_latencies =
          is_custom_execution ? &result->custom_operation_latencies[thread_idx] : nullptr;

      auto thread_affinity_cores = config.thread_pin_mode == ThreadPinMode::AllNumaCores
                                       ? threads_pinning_cores
                                       : CoreIDs{threads_pinning_cores[partition_thread_idx]};

      thread_configs->emplace_back(partition_start, secondary_partition_start, partition_size, secondary_partition_size,
                                   thread_count_per_partition, thread_idx, ops_per_chunk, chunk_count, config,
                                   thread_affinity_cores, execution, total_op_duration, total_op_size,
                                   custom_op_latencies);
    }
  }
}

MemoryRegions Benchmark::prepare_data(const BenchmarkConfig& config) {
  auto region_start_addresses = MemoryRegions(MEM_REGION_COUNT);
  MemaAssert(config.memory_regions.size() == MEM_REGION_COUNT, "Unexpected number of memory regions.");
  // Determines if data shall be written to the memory region so that reads can read the data.
  for (auto region_idx = uint64_t{0}; auto& region_definition : config.memory_regions) {
    if (region_definition.size == 0) {
      region_start_addresses[region_idx] = nullptr;
      ++region_idx;
      continue;
    }
    spdlog::info("Preparing memory region {}.", region_idx);
    switch (region_definition.placement_mode()) {
      case PagePlacementMode::Interleaved:
        region_start_addresses[region_idx] = prepare_interleaved_data(region_definition, config);
        break;
      case PagePlacementMode::Partitioned:
        region_start_addresses[region_idx] = prepare_partitioned_data(region_definition, config);
        break;
      default:
        spdlog::critical("Data preparation mode not handled.");
        utils::crash_exit();
    }
    ++region_idx;
  }
  return region_start_addresses;
}

char* Benchmark::prepare_interleaved_data(const MemoryRegionDefinition& region, const BenchmarkConfig& config) {
  spdlog::info("Preparing interleaved data.");
  auto* data = utils::map(region.size, region.transparent_huge_pages, region.explicit_hugepages_size);
  bind_memory_interleaved(data, region.size, region.node_ids);
  spdlog::debug("Finished mapping memory region.");
  utils::populate_memory(data, region.size);
  spdlog::debug("Finished populating/pre-faulting the memory region.");

  if (config.exec_mode == Mode::DependentReads) {
    utils::generate_shuffled_access_positions(data, region, config);
    spdlog::debug("Finished generating shuffled access positions.");
    if (!utils::verify_shuffled_access_positions(data, region, config)) {
      spdlog::critical("Verifying shuffled access positions failed.");
      utils::crash_exit();
    }
  } else if (config.contains_read_op()) {
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
  spdlog::info("Preparing partitioned data.");
  MemaAssert(region.percentage_pages_first_partition, "Percentage of pages in first partition is not configured.");
  MemaAssert(region.node_count_first_partition, "Number of nodes belonging to the first partition not set.");
  auto* data = utils::map(region.size, region.transparent_huge_pages, region.explicit_hugepages_size);
  spdlog::debug("Finished mapping memory region.");

  const auto region_page_count = region.size / utils::PAGE_SIZE;
  const auto first_partition_page_count =
      static_cast<uint32_t>((*region.percentage_pages_first_partition / 100.f) * region_page_count);

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

  if (config.exec_mode == Mode::DependentReads) {
    utils::generate_shuffled_access_positions(data, region, config);
    spdlog::debug("Finished generating shuffled access positions.");
    if (!utils::verify_shuffled_access_positions(data, region, config)) {
      spdlog::critical("Verifying shuffled access positions failed.");
      utils::crash_exit();
    }
  } else if (config.contains_read_op()) {
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
                                      const MemoryRegionDefinitions& region_definitions) {
  MemaAssert(memory_regions.size() == MEM_REGION_COUNT, "Number of passed memory regions incorrect.");
  MemaAssert(region_definitions.size() == MEM_REGION_COUNT, "Number of passed memory region definitions incorrect.");
  for (auto region_idx = uint64_t{0}; region_idx < MEM_REGION_COUNT; ++region_idx) {
    spdlog::info("Verify page locations of region {}.", region_idx);
    auto verified = false;
    auto& region = memory_regions[region_idx];
    auto& definition = region_definitions[region_idx];
    if (definition.placement_mode() == PagePlacementMode::Interleaved) {
      verified = verify_interleaved_page_placement(region, definition.size, definition.node_ids);
    } else {
      MemaAssert(definition.percentage_pages_first_partition,
                 "Percentage for page placement must be set for partitioned verificatiopn");
      verified = verify_partitioned_page_placement(region, definition.size, definition.node_ids,
                                                   *definition.percentage_pages_first_partition,
                                                   *definition.node_count_first_partition);
    }
    if (!verified) {
      spdlog::critical("Page locations of region {} incorrect.", region_idx);
    }
  }
}

void Benchmark::run_custom_ops_in_thread(ThreadConfig* thread_config, const BenchmarkConfig& config) {
  const auto& operations = config.custom_operations;
  const auto num_ops = operations.size();

  auto operation_chain = std::vector<ChainedOperation>{};
  operation_chain.reserve(num_ops);

  // Determine maximum access size to ensure that operations don't write beyond the end of the range.
  auto max_access_size = uint64_t{0};
  for (const CustomOp& op : operations) {
    max_access_size = std::max(op.size, max_access_size);
  }

  const auto aligned_range_size = uint64_t{thread_config->partition_size - max_access_size};
  const auto aligned_secondary_range_size = uint64_t{thread_config->secondary_partition_size - max_access_size};

  for (auto op_idx = uint64_t{0}; op_idx < num_ops; ++op_idx) {
    const CustomOp& op = operations[op_idx];

    if (op.memory_type == MemoryType::Primary) {
      operation_chain.emplace_back(op, thread_config->start_addr, aligned_range_size);
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

  const auto ops_count_per_chunk = thread_config->ops_count_per_chunk;
  auto total_num_ops = uint64_t{0};

  while (true) {
    if (thread_config->execution->num_custom_chunks_remaining.fetch_sub(1) <= 0) {
      break;
    }

    if (config.latency_sample_frequency == 0) {
      // We don't want the sampling code overhead if we don't want to sample the latency.
      for (size_t iteration = 0; iteration < ops_count_per_chunk; ++iteration) {
        start_op.run(start_addr, start_addr);
      }
    } else {
      // Latency sampling requested, measure the latency every x iterations.
      const uint64_t freq = config.latency_sample_frequency;
      // Start at 1 to avoid measuring latency of first request.
      for (size_t iteration = 1; iteration <= ops_count_per_chunk; ++iteration) {
        if (iteration % freq == 0) {
          auto op_start = std::chrono::steady_clock::now();
          start_op.run(start_addr, start_addr);
          auto op_end = std::chrono::steady_clock::now();
          thread_config->custom_op_latencies->emplace_back((op_end - op_start).count());
        } else {
          start_op.run(start_addr, start_addr);
        }
      }
    }

    total_num_ops += ops_count_per_chunk;
  }

  auto end_ts = std::chrono::steady_clock::now();
  *(thread_config->total_operation_duration) = ExecutionDuration{start_ts, end_ts};
  *(thread_config->total_operation_size) = total_num_ops;
}

template <size_t ACCESS_COUNT_64B>
void Benchmark::run_dependent_reads_in_thread(ThreadConfig* thread_config, const BenchmarkConfig& config) {
  spdlog::debug("Thread {}: Running dependent reads.", thread_config->thread_idx);

  auto buffer = reinterpret_cast<std::byte*>(thread_config->start_addr);
  const auto access_count = thread_config->partition_size / config.access_size;

  auto next_pos = u64{0};
  auto begin_ts = std::chrono::steady_clock::now();

  for (auto op_idx = u64{0}; op_idx < access_count; ++op_idx) {
    auto res =
        rw_ops::read_64B_accesses<ACCESS_COUNT_64B>(reinterpret_cast<char*>(&buffer[next_pos * config.access_size]));
    next_pos = *(reinterpret_cast<u64*>(&res));
  }
  auto end_ts = std::chrono::steady_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_ts - begin_ts);
  spdlog::debug("Thread {}: Finished execution in {} ms", thread_config->thread_idx, duration.count());
  auto average_latency = duration.count() / access_count;
  spdlog::info("Thread {}: Average access latency: {} ns", thread_config->thread_idx, average_latency);

  const auto chunk_size = config.min_io_chunk_size;
  *(thread_config->total_operation_size) = access_count * config.access_size;
  *(thread_config->total_operation_duration) = ExecutionDuration{begin_ts, end_ts};
}

void Benchmark::run_in_thread(ThreadConfig* thread_config, const BenchmarkConfig& config) {
  // Pin thread to the configured numa nodes.
  auto& expected_run_cores = thread_config->affinity_core_ids;
  pin_thread_to_cores(expected_run_cores);
  log_permissions_for_numa_nodes(spdlog::level::debug, thread_config->thread_idx);

  // Check if thread is pinned to the configured NUMA node.
  const auto allowed_cores = allowed_thread_core_ids();
  if (!expected_run_cores.empty() && !utils::is_subset<uint64_t>(allowed_cores, expected_run_cores)) {
    spdlog::critical("Thread #{}: Thread not pinned to the configured CPUs. Expected: [{}], Allowed: [{}]",
                     thread_config->thread_idx, utils::numbers_to_string(expected_run_cores),
                     utils::numbers_to_string(allowed_cores));
    utils::crash_exit();
  }

  if (config.exec_mode == Mode::Custom) {
    return run_custom_ops_in_thread(thread_config, config);
  }

  if (config.exec_mode == Mode::DependentReads) {
    switch (config.access_size) {
      case 64:
        return run_dependent_reads_in_thread<1>(thread_config, config);
      case 128:
        return run_dependent_reads_in_thread<2>(thread_config, config);
      case 256:
        return run_dependent_reads_in_thread<4>(thread_config, config);
      case 512:
        return run_dependent_reads_in_thread<8>(thread_config, config);
      case 1024:
        return run_dependent_reads_in_thread<16>(thread_config, config);
      case 2048:
        return run_dependent_reads_in_thread<32>(thread_config, config);
      case 4096:
        return run_dependent_reads_in_thread<64>(thread_config, config);
      case 8192:
        return run_dependent_reads_in_thread<128>(thread_config, config);
      case 16384:
        return run_dependent_reads_in_thread<256>(thread_config, config);
      case 32768:
        return run_dependent_reads_in_thread<512>(thread_config, config);
      case 65536:
        return run_dependent_reads_in_thread<1024>(thread_config, config);
      default:
        throw MemaException("Dependent Reads only supports specific access sizes between 64 and 65536");
    }
  }

  const size_t seed = std::chrono::steady_clock::now().time_since_epoch().count() * (thread_config->thread_idx + 1);
  lehmer64_seed(seed);

  const uint32_t access_count_in_range = thread_config->partition_size / config.access_size;

  auto access_distribution = [&]() { return lehmer64() % access_count_in_range; };

  spdlog::debug("Thread {}: Starting address generation", thread_config->thread_idx);
  const auto generation_begin_ts = std::chrono::steady_clock::now();

  // Create all chunks before executing.
  size_t chunk_count_per_thread = thread_config->chunk_count / config.number_threads;
  const size_t remaining_chunk_count = thread_config->chunk_count % config.number_threads;
  if (remaining_chunk_count > 0 && thread_config->thread_idx < remaining_chunk_count) {
    // This thread needs to create an extra chunk for an uneven number.
    ++chunk_count_per_thread;
  }

  const size_t thread_idx_in_partition = thread_config->thread_idx % thread_config->thread_count_per_partition;
  const size_t thread_offset_in_partition = thread_idx_in_partition * config.min_io_chunk_size;
  // The size in bytes that all partition threads read. Assuming n chunks C_n and 4 partition threads PT_m, then PT_0,
  // PT_1, PT_2, PT_3 will initally access C_0, C_1, C_2, and C_4 respectively. If PT_0 finished accessing C_0, it will
  // continue with accessing with C_5. The offset delta between C_0's and C_5's start address is 4 * chunk size.
  const size_t partition_threads_access_size = thread_config->thread_count_per_partition * config.min_io_chunk_size;
  const bool is_read_op = config.operation == Operation::Read;

  for (size_t chunk_idx = 0; chunk_idx < chunk_count_per_thread; ++chunk_idx) {
    const size_t thread_chunk_offset =
        // Overall offset after x chunks          + offset of this thread for chunk x+1
        (chunk_idx * partition_threads_access_size) + thread_offset_in_partition;
    char* next_op_position = config.exec_mode == Mode::Sequential_Desc
                                 ? thread_config->start_addr - thread_chunk_offset
                                 : thread_config->start_addr + thread_chunk_offset;

    std::vector<char*> op_addresses(thread_config->ops_count_per_chunk);

    for (size_t op_idx = 0; op_idx < thread_config->ops_count_per_chunk; ++op_idx) {
      switch (config.exec_mode) {
        case Mode::Random: {
          char* partition_start;
          std::function<uint64_t()> random_distribution;
          partition_start = thread_config->start_addr;
          random_distribution = access_distribution;

          uint64_t random_value;
          // Get a random number in the range [0, target_access_count_in_range - 1].
          if (config.random_distribution == RandomDistribution::Uniform) {
            random_value = random_distribution();
          } else {
            random_value = utils::zipf(config.zipf_alpha, access_count_in_range);
          }
          op_addresses[op_idx] = partition_start + (random_value * config.access_size);
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

    // We can always pass the flush_instruction as is. It is ignored for read access.
    Operation op = is_read_op ? Operation::Read : Operation::Write;
    const size_t insert_pos = (chunk_idx * config.number_threads) + thread_config->thread_idx;

    IoOperation& current_op = thread_config->execution->io_operations[insert_pos];
    current_op.op_addresses_ = std::move(op_addresses);
    current_op.access_size_ = config.access_size;
    current_op.op_type_ = op;
    current_op.flush_instruction_ = config.flush_instruction;
  }

  const auto generation_end_ts = std::chrono::steady_clock::now();
  const uint64_t generation_duration_us =
      std::chrono::duration_cast<std::chrono::milliseconds>(generation_end_ts - generation_begin_ts).count();
  spdlog::debug("Thread {}: Finished address generation in {} ms", thread_config->thread_idx, generation_duration_us);

  auto is_last = bool{false};
  uint16_t& threads_remaining = thread_config->execution->threads_remaining;
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
  std::atomic<uint64_t>* io_position = &thread_config->execution->io_position;

  auto executed_op_count = uint64_t{0};
  if (config.run_time == 0) {
    executed_op_count = run_fixed_sized_benchmark(&thread_config->execution->io_operations, io_position);
  } else {
    const auto execution_end = execution_begin_ts + std::chrono::seconds{config.run_time};
    executed_op_count =
        run_duration_based_benchmark(&thread_config->execution->io_operations, io_position, execution_end);
  }

  const auto execution_end_ts = std::chrono::steady_clock::now();
  const auto execution_duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(execution_end_ts - execution_begin_ts);
  spdlog::debug("Thread {}: Finished execution in {} ms", thread_config->thread_idx, execution_duration.count());

  const uint64_t chunk_size = config.access_size * thread_config->ops_count_per_chunk;
  *(thread_config->total_operation_size) = executed_op_count * chunk_size;
  *(thread_config->total_operation_duration) = ExecutionDuration{execution_begin_ts, execution_end_ts};
}

uint64_t Benchmark::run_fixed_sized_benchmark(std::vector<IoOperation>* io_operations,
                                              std::atomic<uint64_t>* io_position) {
  const uint64_t total_op_count = io_operations->size();
  uint64_t executed_op_count = 0;

  while (true) {
    const uint64_t op_pos = io_position->fetch_add(1);
    if (op_pos >= total_op_count) {
      break;
    }

    (*io_operations)[op_pos].run();
    ++executed_op_count;
  }

  return executed_op_count;
}

uint64_t Benchmark::run_duration_based_benchmark(std::vector<IoOperation>* io_operations,
                                                 std::atomic<uint64_t>* io_position,
                                                 std::chrono::steady_clock::time_point execution_end) {
  const uint64_t total_op_count = io_operations->size();
  uint64_t executed_op_count = 0;

  while (true) {
    const uint64_t work_package = io_position->fetch_add(1) % total_op_count;

    (*io_operations)[work_package].run();
    ++executed_op_count;

    const auto current_time = std::chrono::steady_clock::now();
    if (current_time > execution_end) {
      break;
    }
  }

  return executed_op_count;
}

const std::vector<BenchmarkConfig>& Benchmark::get_benchmark_configs() const { return configs_; }

const std::vector<MemoryRegions>& Benchmark::get_memory_regions() const { return memory_regions_; }

const std::vector<std::vector<ThreadConfig>>& Benchmark::get_thread_configs() const { return thread_configs_; }
const std::vector<std::unique_ptr<BenchmarkResult>>& Benchmark::get_benchmark_results() const { return results_; }

nlohmann::json Benchmark::get_json_config(uint8_t config_index) { return configs_[config_index].as_json(); }

void Benchmark::tear_down(bool force) {
  executions_.clear();
  results_.clear();

  for (auto workload_idx = uint64_t{0}; workload_idx < memory_regions_.size(); ++workload_idx) {
    auto& workload_regions = memory_regions_[workload_idx];
    auto& region_definitions = configs_[workload_idx].memory_regions;
    MemaAssert(workload_regions.size() == MEM_REGION_COUNT, "Unexpected memory region count.");
    for (auto region_idx = uint64_t{0}; region_idx < MEM_REGION_COUNT; ++region_idx) {
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

  custom_operation_latencies.clear();
  custom_operation_latencies.shrink_to_fit();
}

nlohmann::json BenchmarkResult::get_result_as_json() const {
  if (config.exec_mode == Mode::Custom) {
    return get_custom_results_as_json();
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

  for (uint16_t thread_idx = 0; thread_idx < config.number_threads; ++thread_idx) {
    const ExecutionDuration& thread_timestamps = total_operation_durations[thread_idx];
    const std::chrono::steady_clock::duration thread_duration = thread_timestamps.duration();
    const auto thread_duration_s = std::chrono::duration<double>(std::chrono::nanoseconds{thread_duration}).count();

    const uint64_t thread_op_size = total_operation_sizes[thread_idx];

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

  nlohmann::json result;
  nlohmann::json bandwidth_results;

  auto execution_time_s = std::chrono::duration<double>(std::chrono::nanoseconds{execution_time});

  bandwidth_results["bandwidth"] = total_bandwidth;
  bandwidth_results["execution_time"] = execution_time_s.count();
  bandwidth_results["accessed_bytes"] = total_size;
  bandwidth_results["thread_bandwidth_avg"] = avg_bandwidth;
  bandwidth_results["thread_bandwidth_std_dev"] = bandwidth_stddev;
  bandwidth_results["thread_op_latency_avg"] = avg_op_latency;
  bandwidth_results["thread_op_latency_std_dev"] = op_latency_stddev;
  bandwidth_results["threads"] = per_thread_results;

  result["results"] = bandwidth_results;

  if (execution_time < std::chrono::seconds{1}) {
    spdlog::warn(
        "Benchmark ran less then 1 second ({} ms). The results may be inaccurate due to the short execution time.",
        std::chrono::duration_cast<std::chrono::milliseconds>(execution_time).count());
  }

  return result;
}

nlohmann::json BenchmarkResult::get_custom_results_as_json() const {
  nlohmann::json custom_op_results;
  nlohmann::json per_thread_results = nlohmann::json::array();

  uint64_t total_num_ops = 0;
  std::vector<double> per_thread_ops_per_s(config.number_threads);

  std::chrono::steady_clock::time_point earliest_begin = total_operation_durations[0].begin;
  std::chrono::steady_clock::time_point latest_end = total_operation_durations[0].end;

  for (uint64_t thread_idx = 0; thread_idx < config.number_threads; ++thread_idx) {
    const ExecutionDuration& thread_timestamps = total_operation_durations[thread_idx];
    const std::chrono::steady_clock::duration thread_duration = thread_timestamps.duration();
    const auto thread_duration_s = std::chrono::duration<double>(std::chrono::nanoseconds{thread_duration}).count();

    const uint64_t num_ops = total_operation_sizes[thread_idx];
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
    for (const std::vector<uint64_t>& thread_latencies : custom_operation_latencies) {
      for (const uint64_t latency : thread_latencies) {
        hdr_record_value(latency_hdr, static_cast<int64_t>(latency));
      }
    }
    custom_op_results["latency"] = hdr_histogram_to_json(latency_hdr);
  }

  nlohmann::json result;
  result["results"] = custom_op_results;
  return result;
}

}  // namespace mema
