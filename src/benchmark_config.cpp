#include "benchmark_config.hpp"

#include <spdlog/spdlog.h>

#include <charconv>
#include <sstream>
#include <string>
#include <unordered_map>

#include "numa.hpp"
#include "utils.hpp"

namespace {

#define CHECK_ARGUMENT(exp, txt)                                          \
  if (!(exp)) {                                                           \
    spdlog::critical(txt + std::string("\nUsed config: ") + to_string()); \
    utils::crash_exit();                                                  \
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
bool get_uints_if_present(YAML::Node& data, const std::string& name, std::vector<T>& values) {
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
    values.push_back(value.as<uint64_t>());
  }
  entry.SetTag(VISITED_TAG);
  return true;
}

}  // namespace

namespace mema {

BenchmarkConfig BenchmarkConfig::decode(YAML::Node& node) {
  spdlog::info("Decoding benchmark config from file: {}", node["config_file"].as<std::string>());
  node.remove("config_file");
  BenchmarkConfig bm_config{};
  size_t found_count = 0;
  try {
    found_count += get_size_if_present(node, "memory_region_size", ConfigEnums::scale_suffix_to_factor,
                                       &bm_config.memory_region_size);
    found_count +=
        get_size_if_present(node, "access_size", ConfigEnums::scale_suffix_to_factor, &bm_config.access_size);
    found_count += get_size_if_present(node, "min_io_chunk_size", ConfigEnums::scale_suffix_to_factor,
                                       &bm_config.min_io_chunk_size);

    found_count += get_if_present(node, "number_operations", &bm_config.number_operations);
    found_count += get_if_present(node, "run_time", &bm_config.run_time);
    found_count += get_if_present(node, "number_partitions", &bm_config.number_partitions);
    found_count += get_if_present(node, "number_threads", &bm_config.number_threads);
    found_count += get_if_present(node, "zipf_alpha", &bm_config.zipf_alpha);
    found_count += get_if_present(node, "prefault_memory", &bm_config.prefault_memory);
    found_count += get_if_present(node, "latency_sample_frequency", &bm_config.latency_sample_frequency);
    found_count += get_if_present(node, "huge_pages", &bm_config.huge_pages);

    found_count += get_enum_if_present(node, "exec_mode", ConfigEnums::str_to_mode, &bm_config.exec_mode);
    found_count += get_enum_if_present(node, "operation", ConfigEnums::str_to_operation, &bm_config.operation);
    found_count += get_enum_if_present(node, "random_distribution", ConfigEnums::str_to_random_distribution,
                                       &bm_config.random_distribution);
    found_count += get_enum_if_present(node, "flush_instruction", ConfigEnums::str_to_flush_instruction,
                                       &bm_config.flush_instruction);
    found_count += get_uints_if_present(node, "numa_memory_nodes", bm_config.numa_memory_nodes);
    found_count += get_uints_if_present(node, "numa_task_nodes", bm_config.numa_task_nodes);

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

  // Check if access size is at least 512-bit, i.e., 64byte (cache line)
  const bool is_access_size_greater_64_byte = access_size >= 64;
  CHECK_ARGUMENT(is_access_size_greater_64_byte, "Access size must be at least 64-byte, i.e., a cache line.");

  // Check if access size is a power of two
  const bool is_access_size_power_of_two = (access_size & (access_size - 1)) == 0;
  CHECK_ARGUMENT(is_access_size_power_of_two, "Access size must be a power of 2.");

  // Check if memory range is multiple of access size
  const bool is_memory_region_size_multiple_of_access_size = (memory_region_size % access_size) == 0;
  CHECK_ARGUMENT(is_memory_region_size_multiple_of_access_size, "Memory range must be a multiple of access size.");

  // Check if at least one thread
  const bool is_at_least_one_thread = number_threads > 0;
  CHECK_ARGUMENT(is_at_least_one_thread, "Number threads must be at least 1.");

  // Assumption: number_threads is multiple of number_partitions
  const bool is_number_threads_multiple_of_number_partitions =
      (number_partitions == 0) || (number_threads % number_partitions) == 0;
  CHECK_ARGUMENT(is_number_threads_multiple_of_number_partitions,
                 "Number threads must be a multiple of number partitions.");

  // Assumption: total memory range must be evenly divisible into number of partitions
  const bool is_partitionable =
      (number_partitions == 0 && ((memory_region_size / number_threads) % access_size) == 0) ||
      (number_partitions > 0 && ((memory_region_size / number_partitions) % access_size) == 0);
  CHECK_ARGUMENT(is_partitionable,
                 "Total memory range must be evenly divisible into number of partitions. "
                 "Most likely you can fix this by using 2^x partitions.");

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
  if (total_accessed_memory < 5 * GIBIBYTES_IN_BYTES) {
    spdlog::warn(
        "Accessing less then 5 GiB of data. This short run may lead to inaccurate results due to the very short "
        "execution.");
  }

  // Assumption: total memory needs to fit into N chunks exactly
  const bool is_time_based_seq_total_memory_chunkable = (memory_region_size % min_io_chunk_size) == 0;
  CHECK_ARGUMENT(is_time_based_seq_total_memory_chunkable,
                 "The total memory range needs to be multiple of chunk size " + std::to_string(min_io_chunk_size));

  // Assumption: we chunk operations, so we need enough data to fill at least one chunk
  const bool is_total_memory_large_enough = (memory_region_size / number_threads) >= min_io_chunk_size;
  CHECK_ARGUMENT(is_total_memory_large_enough,
                 "Each thread needs at least " + std::to_string(min_io_chunk_size) + " Bytes of memory.");

  const bool has_custom_ops = exec_mode != Mode::Custom || !custom_operations.empty();
  CHECK_ARGUMENT(has_custom_ops, "Must specify custom_operations for custom execution.");

  const bool has_no_custom_ops = exec_mode == Mode::Custom || custom_operations.empty();
  CHECK_ARGUMENT(has_no_custom_ops, "Cannot specify custom_operations for non-custom execution.");

  const bool latency_sample_is_custom = exec_mode == Mode::Custom || latency_sample_frequency == 0;
  CHECK_ARGUMENT(latency_sample_is_custom, "Latency sampling can only be used with custom operations.");

  const bool numa_memory_nodes_present = numa_memory_nodes.size() > 0;
  CHECK_ARGUMENT(numa_memory_nodes_present, "NUMA memory nodes must be specified.");

  const bool numa_task_nodes_present = numa_task_nodes.size() > 0;
  CHECK_ARGUMENT(numa_task_nodes_present, "NUMA task nodes must be specified.");

#ifdef HAS_CLWB
  const bool clwb_supported_or_not_used = true;
#else
  const bool clwb_supported_or_not_used = flush_instruction != FlushInstruction::Cache;
#endif
  CHECK_ARGUMENT(clwb_supported_or_not_used, "MemA must be compiled with support for clwb to use clwb flushes.");

#if defined(NT_STORES_AVX_2) || defined(NT_STORES_AVX_512)
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
  stream << "memory range: " << memory_region_size;
  stream << sep << "exec mode: " << utils::get_enum_as_string(ConfigEnums::str_to_mode, exec_mode);
  stream << sep << "memory numa nodes: [";
  auto delim = "";
  for (const auto& node : numa_memory_nodes) {
    stream << delim << node;
    delim = ", ";
  }
  stream << "]" << sep << "task numa nodes: [";
  delim = "";
  for (const auto& node : numa_task_nodes) {
    stream << delim << node;
    delim = ", ";
  }
  stream << "]" << sep << "partition count: " << number_partitions;
  stream << sep << "thread count: " << number_threads;
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

nlohmann::json BenchmarkConfig::as_json() const {
  nlohmann::json config;
  config["memory_region_size"] = memory_region_size;
  config["exec_mode"] = utils::get_enum_as_string(ConfigEnums::str_to_mode, exec_mode);
  config["numa_memory_nodes"] = numa_memory_nodes;
  config["numa_task_nodes"] = numa_task_nodes;
  config["number_partitions"] = number_partitions;
  config["number_threads"] = number_threads;
  config["prefault_memory"] = prefault_memory;
  config["min_io_chunk_size"] = min_io_chunk_size;
  config["huge_pages"] = huge_pages;

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

CustomOp CustomOp::from_string(const std::string& str) {
  if (str.empty()) {
    spdlog::error("Custom operation cannot be empty!");
    utils::crash_exit();
  }

  // Get all parts of the custom operation string representation.
  std::vector<std::string> op_str_parts;
  std::stringstream stream{str};
  std::string op_str_part;
  while (std::getline(stream, op_str_part, '_')) {
    op_str_parts.emplace_back(op_str_part);
  }

  const size_t op_str_part_count = op_str_parts.size();
  if (op_str_part_count < 2) {
    spdlog::error("Custom operation is too short: '{}'. Expected at least <operation>_<size>", str);
    utils::crash_exit();
  }

  // Create new custom operation
  CustomOp custom_op;

  // Get operation and location
  const std::string& operation_str = op_str_parts[0];
  auto op_location_it = ConfigEnums::str_to_operation.find(operation_str);
  if (op_location_it == ConfigEnums::str_to_operation.end()) {
    spdlog::error("Unknown operation: {}", operation_str);
    utils::crash_exit();
  }
  custom_op.type = op_location_it->second;

  // Get size of access
  const std::string& size_str = op_str_parts[1];
  auto size_result = std::from_chars(size_str.data(), size_str.data() + size_str.size(), custom_op.size);
  if (size_result.ec != std::errc()) {
    spdlog::error("Could not parse operation size: {}", size_str);
    utils::crash_exit();
  }

  if ((custom_op.size & (custom_op.size - 1)) != 0) {
    spdlog::error("Access size of custom operation must be power of 2. Got: {}", custom_op.size);
    utils::crash_exit();
  }

  const bool is_write = custom_op.type == Operation::Write;

  if (!is_write) {
    if (op_str_part_count > 2) {
      spdlog::error("Custom read op must not have further information. Got: '{}'", op_str_part_count);
      utils::crash_exit();
    }
    // Read op has no further information.
    return custom_op;
  }

  if (op_str_part_count < 3) {
    spdlog::error("Custom write op must have '_<flush_instruction>' after size, e.g., w64_cache. Got: '{}'", str);
    utils::crash_exit();
  }

  const std::string& flush_str = op_str_parts[2];
  auto flush_it = ConfigEnums::str_to_flush_instruction.find(flush_str);
  if (flush_it == ConfigEnums::str_to_flush_instruction.end()) {
    spdlog::error("Could not parse the flush instruction in write op: '{}'", flush_str);
    utils::crash_exit();
  }

  custom_op.flush = flush_it->second;

  const bool has_offset_information = op_str_part_count == 4;
  if (has_offset_information) {
    const std::string& offset_str = op_str_parts[3];
    auto offset_result = std::from_chars(offset_str.data(), offset_str.data() + offset_str.size(), custom_op.offset);
    if (offset_result.ec != std::errc()) {
      spdlog::error("Could not parse operation offset: {}", offset_str);
      utils::crash_exit();
    }

    const uint64_t absolute_offset = std::abs(custom_op.offset);
    if ((absolute_offset % 64) != 0) {
      spdlog::error("Offset of custom write operation must be multiple of 64. Got: {}", custom_op.offset);
      utils::crash_exit();
    }
  }

  return custom_op;
}

std::vector<CustomOp> CustomOp::all_from_string(const std::string& str) {
  if (str.empty()) {
    spdlog::error("Custom operations cannot be empty!");
    utils::crash_exit();
  }

  std::vector<CustomOp> ops;
  std::stringstream stream{str};
  std::string op_str;
  while (std::getline(stream, op_str, ',')) {
    ops.emplace_back(from_string(op_str));
  }

  // Check if operation chain is valid
  const bool is_valid = validate(ops);
  if (!is_valid) {
    spdlog::error("Got invalid custom operations: {}", str);
    utils::crash_exit();
  }

  return ops;
}

std::string CustomOp::to_string() const {
  std::stringstream out;
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

bool CustomOp::validate(const std::vector<CustomOp>& operations) {
  if (operations[0].type != Operation::Read) {
    spdlog::error("First custom operation must be a read");
    return false;
  }

  return true;
}

bool CustomOp::operator==(const CustomOp& rhs) const {
  return type == rhs.type && size == rhs.size && flush == rhs.flush && offset == rhs.offset;
}
bool CustomOp::operator!=(const CustomOp& rhs) const { return !(rhs == *this); }
std::ostream& operator<<(std::ostream& os, const CustomOp& op) { return os << op.to_string(); }

const std::unordered_map<std::string, Mode> ConfigEnums::str_to_mode{{"sequential", Mode::Sequential},
                                                                     {"sequential_desc", Mode::Sequential_Desc},
                                                                     {"random", Mode::Random},
                                                                     {"custom", Mode::Custom}};

const std::unordered_map<std::string, Operation> ConfigEnums::str_to_operation{
    {"read", Operation::Read}, {"write", Operation::Write}, {"r", Operation::Read}, {"w", Operation::Write}};

const std::unordered_map<std::string, FlushInstruction> ConfigEnums::str_to_flush_instruction{
    {"nocache", FlushInstruction::NoCache}, {"cache", FlushInstruction::Cache}, {"none", FlushInstruction::None}};

const std::unordered_map<std::string, RandomDistribution> ConfigEnums::str_to_random_distribution{
    {"uniform", RandomDistribution::Uniform}, {"zipf", RandomDistribution::Zipf}};

const std::unordered_map<char, uint64_t> ConfigEnums::scale_suffix_to_factor{{'k', 1024},
                                                                             {'K', 1024},
                                                                             {'m', 1024 * 1024},
                                                                             {'M', 1024 * 1024},
                                                                             {'g', 1024 * 1024 * 1024},
                                                                             {'G', 1024 * 1024 * 1024}};

}  // namespace mema
