#include "benchmark_config.hpp"

#include <spdlog/spdlog.h>

#include <charconv>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>

#include "numa.hpp"
#include "utils.hpp"

namespace {

#define CHECK_ARGUMENT(exp, txt)                                          \
  if (!(exp)) {                                                           \
    spdlog::critical(txt + std::string("\nUsed config: ") + to_string()); \
    spdlog::default_logger()->flush();                                    \
    utils::crash_exit(txt);                                               \
  }                                                                       \
  static_assert(true, "Need ; after macro")

constexpr auto VISITED_TAG = "visited";

void ensure_unique_key(const YAML::Node& entry, const std::string& name) {
  if (entry.Tag() == VISITED_TAG) {
    const YAML::Mark& mark = entry.Mark();
    spdlog::critical("Duplicate entry: '{}' (in line: {})", mark.line, name);
    mema::utils::crash_exit();
  }
}

template <typename T>
bool get_optional_if_present(YAML::Node& data, const std::string& name, std::optional<T>* attribute) {
  YAML::Node entry = data[name];
  if (!entry) {
    return false;
  }
  ensure_unique_key(entry, name);

  *attribute = entry.as<T>();

  entry.SetTag(VISITED_TAG);
  return true;
}

template <typename T>
bool get_if_present(YAML::Node& data, const std::string& name, T* attribute) {
  YAML::Node entry = data[name];
  if (!entry) {
    return false;
  }
  ensure_unique_key(entry, name);

  *attribute = entry.as<T>();

  entry.SetTag(VISITED_TAG);
  return true;
}

template <typename T>
bool get_enum_if_present(YAML::Node& data, const std::string& name, const std::unordered_map<std::string, T>& enum_map,
                         T* attribute) {
  YAML::Node entry = data[name];
  if (!entry) {
    return false;
  }
  ensure_unique_key(entry, name);

  const auto enum_key = entry.as<std::string>();
  auto it = enum_map.find(enum_key);
  if (it == enum_map.end()) {
    spdlog::critical("Unknown '{}': {}", name, enum_key);
    mema::utils::crash_exit();
  }

  *attribute = it->second;
  entry.SetTag(VISITED_TAG);
  return true;
}

template <typename T>
bool get_size_if_present(YAML::Node& data, const std::string& name, const std::unordered_map<char, uint64_t>& enum_map,
                         T* attribute) {
  YAML::Node entry = data[name];
  if (!entry) {
    return false;
  }
  ensure_unique_key(entry, name);

  const auto size_string = entry.as<std::string>();
  const char size_suffix = size_string.back();
  size_t size_end = size_string.length();
  uint64_t factor = 1;

  auto it = enum_map.find(size_suffix);
  if (it != enum_map.end()) {
    factor = it->second;
    size_end -= 1;
  } else if (isalpha(size_suffix)) {
    spdlog::critical("Unknown size suffix: {}", size_suffix);
    mema::utils::crash_exit();
  }

  char* end;
  const std::string size_number = size_string.substr(0, size_end);
  const uint64_t size = std::strtoull(size_number.data(), &end, 10);
  *attribute = size * factor;

  entry.SetTag(VISITED_TAG);
  return true;
}

template <typename T>
bool get_sequence_if_present(YAML::Node& data, const std::string& name, std::vector<T>& values) {
  auto entry = data[name];
  if (!entry) {
    return false;
  }

  if (!entry.IsSequence()) {
    spdlog::critical("Value of key {} must be a YAML sequence, i.e., [0, 2, 3].", name);
    mema::utils::crash_exit();
  }

  values.reserve(entry.size());
  for (const auto value : entry) {
    values.push_back(value.as<T>());
  }
  entry.SetTag(VISITED_TAG);
  return true;
}

}  // namespace

namespace mema {

PagePlacementMode MemoryRegionDefinition::placement_mode() const {
  if (percentage_pages_first_partition) {
    return PagePlacementMode::Partitioned;
  }
  return PagePlacementMode::Interleaved;
}

BenchmarkConfig BenchmarkConfig::decode(YAML::Node& node) {
  spdlog::debug("Decoding benchmark config from file: {}", node["config_file"].as<std::string>());
  node.remove("config_file");
  BenchmarkConfig bm_config{};
  size_t found_count = 0;
  try {
    // Memory region related
    found_count += get_size_if_present(node, "memory_region_size", ConfigEnums::scale_suffix_to_factor,
                                       &bm_config.memory_regions[0].size);
    found_count += get_size_if_present(node, "secondary_memory_region_size", ConfigEnums::scale_suffix_to_factor,
                                       &bm_config.memory_regions[1].size);
    found_count += get_sequence_if_present(node, "numa_memory_nodes", bm_config.memory_regions[0].node_ids);
    found_count += get_sequence_if_present(node, "secondary_numa_memory_nodes", bm_config.memory_regions[1].node_ids);
    found_count += get_if_present(node, "transparent_huge_pages", &bm_config.memory_regions[0].transparent_huge_pages);
    found_count +=
        get_if_present(node, "secondary_transparent_huge_pages", &bm_config.memory_regions[1].transparent_huge_pages);
    found_count += get_size_if_present(node, "explicit_hugepages_size", ConfigEnums::scale_suffix_to_factor,
                                       &bm_config.memory_regions[0].explicit_hugepages_size);
    found_count += get_size_if_present(node, "secondary_explicit_hugepages_size", ConfigEnums::scale_suffix_to_factor,
                                       &bm_config.memory_regions[1].explicit_hugepages_size);
    // We only set the percentage for the primary memory region
    found_count += get_optional_if_present(node, "percentage_pages_first_partition",
                                           &bm_config.memory_regions[0].percentage_pages_first_partition);
    found_count += get_optional_if_present(node, "node_count_first_partition",
                                           &bm_config.memory_regions[0].node_count_first_partition);
    found_count += get_if_present(node, "number_partitions", &bm_config.number_partitions);

    found_count +=
        get_size_if_present(node, "access_size", ConfigEnums::scale_suffix_to_factor, &bm_config.access_size);
    found_count += get_size_if_present(node, "min_io_chunk_size", ConfigEnums::scale_suffix_to_factor,
                                       &bm_config.min_io_chunk_size);
    found_count += get_if_present(node, "number_operations", &bm_config.number_operations);
    found_count += get_if_present(node, "run_time", &bm_config.run_time);
    found_count += get_if_present(node, "number_threads", &bm_config.number_threads);
    found_count += get_if_present(node, "zipf_alpha", &bm_config.zipf_alpha);
    found_count += get_if_present(node, "latency_sample_frequency", &bm_config.latency_sample_frequency);
    found_count += get_enum_if_present(node, "exec_mode", ConfigEnums::str_to_mode, &bm_config.exec_mode);
    found_count += get_enum_if_present(node, "operation", ConfigEnums::str_to_operation, &bm_config.operation);
    found_count += get_enum_if_present(node, "random_distribution", ConfigEnums::str_to_random_distribution,
                                       &bm_config.random_distribution);
    found_count += get_enum_if_present(node, "flush_instruction", ConfigEnums::str_to_flush_instruction,
                                       &bm_config.flush_instruction);
    found_count += get_sequence_if_present(node, "numa_task_nodes", bm_config.numa_thread_nodes);
    found_count +=
        get_enum_if_present(node, "thread_pin_mode", ConfigEnums::str_to_thread_pin_mode, &bm_config.thread_pin_mode);
    found_count += get_sequence_if_present(node, "thread_cores", bm_config.thread_core_ids);

    std::string custom_ops;
    const bool has_custom_ops = get_if_present(node, "custom_operations", &custom_ops);
    if (has_custom_ops) {
      bm_config.custom_operations = CustomOp::all_from_string(custom_ops);
      ++found_count;
    }

    if (found_count != node.size()) {
      for (YAML::const_iterator entry = node.begin(); entry != node.end(); ++entry) {
        if (entry->second.Tag() != VISITED_TAG) {
          spdlog::critical("Unknown config entry '{}' in line: {}", entry->first.as<std::string>(),
                           std::to_string(entry->second.Mark().line));
          utils::crash_exit();
        }
      }
    }
  } catch (const YAML::InvalidNode& e) {
    spdlog::critical("Exception during config parsing: {}", e.msg);
    utils::crash_exit();
  }

  bm_config.validate();
  return bm_config;
}

void BenchmarkConfig::validate() const {
  bool is_custom_or_random = exec_mode == Mode::Random || exec_mode == Mode::Custom;

  // Check if access size is supported
  std::set<size_t> unrolled_access_sizes({64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536});
  std::ostringstream error;
  error << "Access Size must be one of {";
  for (auto it = unrolled_access_sizes.begin(); it != unrolled_access_sizes.end(); ++it) {
    error << *it << ((std::next(it) != unrolled_access_sizes.end()) ? ", " : "}");
  }
  error << "or greater 65536.";

  CHECK_ARGUMENT(unrolled_access_sizes.contains(access_size) || (access_size > 65536), error.str());

  // Check if at least one thread
  const bool is_at_least_one_thread = number_threads > 0;
  CHECK_ARGUMENT(is_at_least_one_thread, "Number threads must be at least 1.");

  // Memory region checks
  for (auto& region : memory_regions) {
    // Check if memory range is multiple of access size
    const bool is_memory_region_size_multiple_of_access_size = (region.size % access_size) == 0;
    CHECK_ARGUMENT(is_memory_region_size_multiple_of_access_size, "Memory range must be a multiple of access size.");

    // Check if NUMA memory nodes are specified for the memory region.
    const bool numa_memory_nodes_present = region.size == 0 || region.node_ids.size() > 0;
    CHECK_ARGUMENT(numa_memory_nodes_present, "NUMA memory nodes must be specified.");

    // Check if share of pages on first node has a valid value if set.
    if (region.percentage_pages_first_partition) {
      CHECK_ARGUMENT(region.node_count_first_partition.has_value(),
                     "Partitioned mode requires node_count_first_partition, which is not set.");
      CHECK_ARGUMENT(*region.node_count_first_partition <= region.node_ids.size(),
                     "node_count_first_partition must not be larger than the number of specified NUMA node IDs.");
      const bool has_memory_share_correct_value = *region.percentage_pages_first_partition <= 100;
      CHECK_ARGUMENT(has_memory_share_correct_value, "Share of pages located on first node must be in range [0, 100].");
      const bool has_gte_two_numa_memory_nodes = region.node_ids.size() >= 2;
      CHECK_ARGUMENT(has_gte_two_numa_memory_nodes,
                     "When a share of pages located on first node is specified, >=2 nodes need to be specified.");
    }

    // Assumption: total memory range must be evenly divisible into number of partitions
    const bool is_partitionable = (number_partitions == 0 && ((region.size / number_threads) % access_size) == 0) ||
                                  (number_partitions > 0 && ((region.size / number_partitions) % access_size) == 0);
    CHECK_ARGUMENT(is_partitionable,
                   "Total memory range must be evenly divisible into number of partitions. "
                   "Most likely you can fix this by using 2^x partitions.");
  }

  const bool has_secondary_memory_region_for_operations = !contains_secondary_memory_op() || memory_regions[1].size > 0;
  CHECK_ARGUMENT(has_secondary_memory_region_for_operations,
                 "Must set secondary_memory_region_size > 0 if the benchmark contains secondary memory operations.");

  // Assumption: total memory needs to fit into N chunks exactly
  const bool is_total_memory_chunkable = (memory_regions[0].size % min_io_chunk_size) == 0;
  CHECK_ARGUMENT(is_total_memory_chunkable,
                 "The primary memory range needs to be multiple of chunk size " + std::to_string(min_io_chunk_size));

  if (exec_mode != Mode::DependentReads) {
    // Assumption: we chunk operations, so we need enough data to fill at least one chunk
    const bool is_total_memory_large_enough = (memory_regions[0].size / number_threads) >= min_io_chunk_size;
    CHECK_ARGUMENT(is_total_memory_large_enough, "Each thread needs at least " + std::to_string(min_io_chunk_size) +
                                                     " Bytes of memory in primary region.");
  }

  // Assumption: number_threads is multiple of number_partitions
  const bool is_number_threads_multiple_of_number_partitions =
      (number_partitions == 0) || (number_threads % number_partitions) == 0;
  CHECK_ARGUMENT(is_number_threads_multiple_of_number_partitions,
                 "Number threads must be a multiple of number partitions.");

  // Assumption: number_operations should only be set for random/custom access. It is ignored in sequential IO.
  const bool is_number_operations_set_random =
      number_operations == BenchmarkConfig{}.number_operations || is_custom_or_random;
  CHECK_ARGUMENT(is_number_operations_set_random, "Number of operations should only be set for random/custom access.");

  // Assumption: min_io_chunk size must be a power of two
  const bool is_valid_min_io_chunk_size = min_io_chunk_size >= 64 && (min_io_chunk_size & (min_io_chunk_size - 1)) == 0;
  CHECK_ARGUMENT(is_valid_min_io_chunk_size, "Minimum IO chunk must be >= 64 Byte and a power of two.");

  // Assumption: We need enough operations to give each thread at least one chunk
  const uint64_t min_required_number_ops = (min_io_chunk_size / access_size) * number_threads;
  const bool has_enough_number_operations = !is_custom_or_random || number_operations >= min_required_number_ops;
  CHECK_ARGUMENT(has_enough_number_operations,
                 "Need enough number_operations to have at least one chunk per thread. Consider at least 100 "
                 "operations in total to actually perform a significant amount of work. Need minimum of " +
                     std::to_string(min_required_number_ops) + " ops for this workload.");

  const uint64_t total_accessed_memory = number_operations * access_size;
  if (total_accessed_memory < 5 * GiB) {
    spdlog::warn(
        "Accessing less then 5 GiB of data. This short run may lead to inaccurate results due to the very short "
        "execution.");
  }

  const bool has_custom_ops = exec_mode != Mode::Custom || !custom_operations.empty();
  CHECK_ARGUMENT(has_custom_ops, "Must specify custom_operations for custom execution.");

  const bool has_no_custom_ops = exec_mode == Mode::Custom || custom_operations.empty();
  CHECK_ARGUMENT(has_no_custom_ops, "Cannot specify custom_operations for non-custom execution.");

  const bool latency_sample_is_custom = exec_mode == Mode::Custom || latency_sample_frequency == 0;
  CHECK_ARGUMENT(latency_sample_is_custom, "Latency sampling can only be used with custom operations.");

  const bool numa_thread_nodes_present = numa_thread_nodes.size() > 0;
  const bool numa_thread_pinning_mode =
      (thread_pin_mode == ThreadPinMode::AllNumaCores || thread_pin_mode == ThreadPinMode::SingleNumaCoreIncrement);
  CHECK_ARGUMENT(numa_thread_nodes_present || !numa_thread_pinning_mode,
                 "NUMA task nodes must be specified with a NUMA-specific thread pinning mode.");
  CHECK_ARGUMENT(numa_thread_pinning_mode || !thread_core_ids.empty(),
                 "Core IDs must be specified if thread pinning is not NUMA-specific.");

  if (thread_pin_mode == ThreadPinMode::SingleCoreFixed) {
    CHECK_ARGUMENT(thread_core_ids.size() == number_threads, "Number of Core IDs and thread count must be equal.");
  }

#ifdef HAS_CLWB
  const bool clwb_supported_or_not_used = true;
#else
  const bool clwb_supported_or_not_used = flush_instruction != FlushInstruction::Cache;
#endif
  CHECK_ARGUMENT(clwb_supported_or_not_used, "MemA must be compiled with support for clwb to use clwb flushes.");

#if defined(USE_AVX_2) || defined(USE_AVX_512)
  const bool nt_stores_supported_or_not_used = true;
#else
  const bool nt_stores_supported_or_not_used = flush_instruction != FlushInstruction::NoCache;
#endif
  CHECK_ARGUMENT(nt_stores_supported_or_not_used, "MemA must be compiled with support for NT stores to use NT stores.");
}

bool BenchmarkConfig::contains_read_op() const { return operation == Operation::Read || exec_mode == Mode::Custom; }

bool BenchmarkConfig::contains_write_op() const {
  auto find_custom_write_op = [](const CustomOp& op) { return op.type == Operation::Write; };
  return operation == Operation::Write ||
         std::any_of(custom_operations.begin(), custom_operations.end(), find_custom_write_op);
}

std::string BenchmarkConfig::to_string(const std::string sep) const {
  auto stream = std::stringstream{};
  for (auto region_idx = uint64_t{0}; auto& region : memory_regions) {
    if (region.size == 0) {
      continue;
    }
    if (region_idx > 0) {
      stream << sep;
    }
    stream << "memory region " << region_idx << " size: " << region.size;
    stream << sep << "numa nodes: [";
    auto delim = "";
    for (const auto& node : region.node_ids) {
      stream << delim << node;
      delim = ", ";
    }
    stream << "]";
    stream << sep << "partition count: " << number_partitions;
    stream << sep << "page placement: ";
    if (region.percentage_pages_first_partition) {
      stream << "partitioned with " << *region.percentage_pages_first_partition << "% on first partition. ";
      stream << "The first " << *region.node_count_first_partition << "nodes belong to the first partition.";
    } else {
      stream << "interleaved";
    }
    ++region_idx;
  }
  stream << sep << "exec mode: " << utils::get_enum_as_string(ConfigEnums::str_to_mode, exec_mode);
  stream << sep << "thread numa nodes: [" << utils::numbers_to_string(numa_thread_nodes) << "]";
  stream << sep << "thread count: " << number_threads;
  stream << sep
         << "thread pinning: " << utils::get_enum_as_string(ConfigEnums::str_to_thread_pin_mode, thread_pin_mode);
  stream << sep << "thread core IDs: [" << utils::numbers_to_string(thread_core_ids) << "]";
  stream << sep << "min io chunk size: " << min_io_chunk_size;

  if (exec_mode != Mode::Custom) {
    stream << sep << "access size: " << access_size;
    stream << sep << "operation: " << utils::get_enum_as_string(ConfigEnums::str_to_operation, operation);

    if (operation == Operation::Write) {
      stream << sep << "flush instruction: "
             << utils::get_enum_as_string(ConfigEnums::str_to_flush_instruction, flush_instruction);
    }
  }

  if (exec_mode == Mode::Random) {
    stream << sep << "number operations: " << number_operations;
    stream << sep << "random distribution: "
           << utils::get_enum_as_string(ConfigEnums::str_to_random_distribution, random_distribution);
    if (random_distribution == RandomDistribution::Zipf) {
      stream << sep << "zipf alpha: " << zipf_alpha;
    }
  }

  if (exec_mode == Mode::Custom) {
    stream << sep << "number operations: " << number_operations;
    stream << sep << "custom operations: " << CustomOp::all_to_string(custom_operations);
  }

  if (run_time > 0) {
    stream << sep << "run time: " << run_time;
  }

  return stream.str();
}

bool BenchmarkConfig::contains_secondary_memory_op() const {
  auto find_custom_secondary_memory_op = [](const CustomOp& op) { return op.memory_type == MemoryType::Secondary; };
  return std::any_of(custom_operations.begin(), custom_operations.end(), find_custom_secondary_memory_op);
}

nlohmann::json BenchmarkConfig::as_json() const {
  nlohmann::json config;
  for (auto region_idx = uint64_t{0}; auto& region : memory_regions) {
    auto prefix = MEM_REGION_PREFIX + std::to_string(region_idx) + "_";
    config[prefix + "explicit_hugepages_size"] = region.explicit_hugepages_size;
    config[prefix + "region_size"] = region.size;
    config[prefix + "numa_nodes"] = region.node_ids;
    config[prefix + "percentage_pages_first_partition"] =
        region.percentage_pages_first_partition ? *region.percentage_pages_first_partition : -1;
    config[prefix + "node_count_first_partition"] =
        region.node_count_first_partition ? *region.node_count_first_partition : -1;
    config[prefix + "transparent_huge_pages"] = region.transparent_huge_pages;
    ++region_idx;
  }
  config["exec_mode"] = utils::get_enum_as_string(ConfigEnums::str_to_mode, exec_mode);
  config["min_io_chunk_size"] = min_io_chunk_size;
  config["numa_task_nodes"] = numa_thread_nodes;
  config["number_partitions"] = number_partitions;
  config["number_threads"] = number_threads;
  config["thread_pin_mode"] = utils::get_enum_as_string(ConfigEnums::str_to_thread_pin_mode, thread_pin_mode);
  config["thread_cores"] = thread_core_ids;

  if (exec_mode != Mode::Custom) {
    config["access_size"] = access_size;
    config["operation"] = utils::get_enum_as_string(ConfigEnums::str_to_operation, operation);

    if (operation == Operation::Write) {
      config["flush_instruction"] = utils::get_enum_as_string(ConfigEnums::str_to_flush_instruction, flush_instruction);
    }
  }

  if (exec_mode == Mode::Random) {
    config["number_operations"] = number_operations;
    config["random_distribution"] =
        utils::get_enum_as_string(ConfigEnums::str_to_random_distribution, random_distribution);
    if (random_distribution == RandomDistribution::Zipf) {
      config["zipf_alpha"] = zipf_alpha;
    }
  }

  if (exec_mode == Mode::Custom) {
    config["number_operations"] = number_operations;
    config["custom_operations"] = CustomOp::all_to_string(custom_operations);
  }

  if (run_time > 0) {
    config["run_time"] = run_time;
  }

  return config;
}

uint64_t CustomOp::cumulative_size(const std::vector<CustomOp>& ops) {
  auto size = uint64_t{0};
  for (const auto& op : ops) {
    size += op.size;
  }
  return size;
}

CustomOp CustomOp::from_string(const std::string& str) {
  if (str.empty()) {
    spdlog::critical("Custom operation cannot be empty!");
    utils::crash_exit();
  }

  // Get all parts of the custom operation string representation.
  std::vector<std::string> op_str_parts;
  std::stringstream stream{str};
  std::string op_str_part;
  while (std::getline(stream, op_str_part, '_')) {
    op_str_parts.emplace_back(op_str_part);
  }

  // Determine the offset where
  auto has_memory_region_idx = str[0] == MEM_REGION_PREFIX;

  const size_t op_str_part_count = op_str_parts.size();
  if (op_str_part_count < 2 + has_memory_region_idx) {
    spdlog::critical("Custom operation is too short: '{}'. Expected at least <operation>_<size>", str);
    utils::crash_exit();
  }

  // Create new custom operation
  CustomOp custom_op;

  // Get memory region if set
  if (has_memory_region_idx) {
    const auto region_idx_str = op_str_parts[0].erase(0, 1);
    const auto region_idx = std::stoul(region_idx_str);
    if (region_idx > 0) {
      spdlog::debug("Custom operation with memory region index {}.", region_idx);
      custom_op.memory_type = MemoryType::Secondary;
    }
  }

  // Get operation and location
  const std::string& operation_str = op_str_parts[0 + has_memory_region_idx];
  auto op_location_it = ConfigEnums::str_to_operation.find(operation_str);
  if (op_location_it == ConfigEnums::str_to_operation.end()) {
    spdlog::critical("Unknown operation: {}", operation_str);
    utils::crash_exit();
  }
  custom_op.type = op_location_it->second;

  // Get size of access
  const std::string& size_str = op_str_parts[1 + has_memory_region_idx];
  auto size_result = std::from_chars(size_str.data(), size_str.data() + size_str.size(), custom_op.size);
  if (size_result.ec != std::errc()) {
    spdlog::critical("Could not parse operation size: {}", size_str);
    utils::crash_exit();
  }

  if ((custom_op.size & (custom_op.size - 1)) != 0) {
    spdlog::critical("Access size of custom operation must be power of 2. Got: {}", custom_op.size);
    utils::crash_exit();
  }

  const bool is_write = custom_op.type == Operation::Write;

  if (!is_write) {
    if (op_str_part_count > 2 + has_memory_region_idx) {
      spdlog::critical("Custom read op must not have further information. Got: '{}'", op_str_part_count);
      utils::crash_exit();
    }
    // Read op has no further information.
    return custom_op;
  }

  if (op_str_part_count < 3 + has_memory_region_idx) {
    spdlog::critical("Custom write op must have '_<flush_instruction>' after size, e.g., w64_cache. Got: '{}'", str);
    utils::crash_exit();
  }

  const std::string& flush_str = op_str_parts[2 + has_memory_region_idx];
  auto flush_it = ConfigEnums::str_to_flush_instruction.find(flush_str);
  if (flush_it == ConfigEnums::str_to_flush_instruction.end()) {
    spdlog::critical("Could not parse the flush instruction in write op: '{}'", flush_str);
    utils::crash_exit();
  }

  custom_op.flush = flush_it->second;

  const bool has_offset_information = op_str_part_count == 4 + has_memory_region_idx;
  if (has_offset_information) {
    const std::string& offset_str = op_str_parts[3 + has_memory_region_idx];
    auto offset_result = std::from_chars(offset_str.data(), offset_str.data() + offset_str.size(), custom_op.offset);
    if (offset_result.ec != std::errc()) {
      spdlog::critical("Could not parse operation offset: {}", offset_str);
      utils::crash_exit();
    }

    const uint64_t absolute_offset = std::abs(custom_op.offset);
    if ((absolute_offset % 64) != 0) {
      spdlog::critical("Offset of custom write operation must be multiple of 64. Got: {}", custom_op.offset);
      utils::crash_exit();
    }
  }

  return custom_op;
}

std::vector<CustomOp> CustomOp::all_from_string(const std::string& str) {
  if (str.empty()) {
    spdlog::critical("Custom operations cannot be empty!");
    utils::crash_exit();
  }

  auto ops = std::vector<CustomOp>{};
  auto stream = std::stringstream{str};
  auto op_str = std::string{};
  while (std::getline(stream, op_str, ',')) {
    ops.emplace_back(from_string(op_str));
  }

  // Check if operation chain is valid
  validate(ops);

  return ops;
}

std::string CustomOp::to_string() const {
  std::stringstream out;
  out << MEM_REGION_PREFIX << (memory_type == MemoryType::Primary ? 0 : 1) << '_';
  out << utils::get_enum_as_string(ConfigEnums::str_to_operation, type, true);
  out << '_' << size;
  if (type == Operation::Write) {
    out << '_' << utils::get_enum_as_string(ConfigEnums::str_to_flush_instruction, flush);
    if (offset != 0) {
      out << '_' << offset;
    }
  }
  return out.str();
}

std::string CustomOp::to_string(const CustomOp& op) { return op.to_string(); }

std::string CustomOp::all_to_string(const std::vector<CustomOp>& ops) {
  if (ops.empty()) {
    return "";
  }

  std::stringstream out;
  for (size_t i = 0; i < ops.size() - 1; ++i) {
    out << to_string(ops[i]) << ',';
  }
  out << to_string(ops[ops.size() - 1]);
  return out.str();
}

void CustomOp::validate(const std::vector<CustomOp>& operations) {
  if (operations[0].type != Operation::Read) {
    spdlog::critical("First custom operation must be a read");
    utils::crash_exit();
  }

  // Check if write is to same memory region type
  auto current_memory_type = operations[0].memory_type;
  for (const CustomOp& op : operations) {
    if ((op.type == Operation::Write) && (current_memory_type != op.memory_type)) {
      spdlog::critical("A write must occur after a read to the same memory type.");
      spdlog::critical("Bad operation: {}", op.to_string());
      utils::crash_exit();
    }
    current_memory_type = op.memory_type;
  }
}

bool CustomOp::operator==(const CustomOp& rhs) const {
  return type == rhs.type && size == rhs.size && flush == rhs.flush && offset == rhs.offset;
}
bool CustomOp::operator!=(const CustomOp& rhs) const { return !(rhs == *this); }
std::ostream& operator<<(std::ostream& os, const CustomOp& op) { return os << op.to_string(); }

const std::unordered_map<std::string, Mode> ConfigEnums::str_to_mode{{"sequential", Mode::Sequential},
                                                                     {"sequential_desc", Mode::Sequential_Desc},
                                                                     {"random", Mode::Random},
                                                                     {"custom", Mode::Custom},
                                                                     {"dependent_reads", Mode::DependentReads}};

const std::unordered_map<std::string, Operation> ConfigEnums::str_to_operation{
    {"read", Operation::Read}, {"write", Operation::Write}, {"r", Operation::Read}, {"w", Operation::Write}};

const std::unordered_map<std::string, FlushInstruction> ConfigEnums::str_to_flush_instruction{
    {"nocache", FlushInstruction::NoCache}, {"cache", FlushInstruction::Cache}, {"none", FlushInstruction::None}};

const std::unordered_map<std::string, RandomDistribution> ConfigEnums::str_to_random_distribution{
    {"uniform", RandomDistribution::Uniform}, {"zipf", RandomDistribution::Zipf}};

const std::unordered_map<std::string, ThreadPinMode> ConfigEnums::str_to_thread_pin_mode{
    {"all-numa", ThreadPinMode::AllNumaCores},
    {"single-numa", ThreadPinMode::SingleNumaCoreIncrement},
    {"single-fixed", ThreadPinMode::SingleCoreFixed}};

const std::unordered_map<char, uint64_t> ConfigEnums::scale_suffix_to_factor{{'k', 1024},
                                                                             {'K', 1024},
                                                                             {'m', 1024 * 1024},
                                                                             {'M', 1024 * 1024},
                                                                             {'g', 1024 * 1024 * 1024},
                                                                             {'G', 1024 * 1024 * 1024}};

}  // namespace mema
