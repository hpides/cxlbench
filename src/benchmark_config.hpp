#pragma once

#include <array>
#include <optional>
#include <ostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "json.hpp"
#include "types.hpp"
#include "yaml-cpp/yaml.h"

namespace mema {

enum class Mode : u8 { Sequential, Sequential_Desc, Random, Custom, DependentReads };

enum class RandomDistribution : u8 { Uniform, Zipf };

enum class CacheInstruction : u8 { Cache, NoCache, None };

enum class Operation : u8 { Read, Write };

enum class MemoryType : u8 { Primary, Secondary };

enum class PagePlacementMode : u8 { Interleaved, Partitioned };

// AllNumaCores pins a thread to the set of cores of a given NUMA node.
// SingleNumaCoreIncrement pins each thread to a single core, which is in the set of cores of a given NUMA node. The
// tool automatically determines the next available core when configuring the threads.
// SingleCoreFixed pints each thread to a single core. This is fully user-defined. The cores to pin threads at need to
// match the thread count for the workload.
enum class ThreadPinMode : u8 { AllNumaCores, SingleNumaCoreIncrement, SingleCoreFixed };

static constexpr size_t MiB = 1024u * 1024;
static constexpr size_t GiB = 1024u * MiB;
static constexpr size_t SECONDS_IN_NANOSECONDS = 1e9;

constexpr auto MEM_REGION_COUNT = u64{2};
constexpr auto PAR_WORKLOAD_COUNT = u64{2};
constexpr auto MEM_REGION_PREFIX = 'm';

/**
 * This represents a custom operation to be specified by the user. Its string representation, is:
 *
 * For reads: r_<size>
 *
 * with:
 * 'r' for read,
 * <size> is the size of the access (must be power of 2).
 *
 * For writes: w_<size>_<cache_instruction>(_<offset>)
 *
 * with:
 * 'w' for write,
 * <size> is the size of the access (must be power of 2),
 * <cache_instruction> is the instruction to use after the write (none, cache, noache),
 * (optional) <offset> is the offset to the previously accessed address (can be negative, default is 0)
 *
 * */
struct CustomOp {
  MemoryType memory_type = MemoryType::Primary;
  Operation type;
  u64 size;
  CacheInstruction cache_fn = CacheInstruction::None;
  // This can be signed, e.g., to represent the case when the previous cache line should be written to.
  i64 offset = 0;

  static CustomOp from_string(const std::string& str);
  static std::vector<CustomOp> all_from_string(const std::string& str);
  static std::string all_to_string(const std::vector<CustomOp>& ops);
  static std::string to_string(const CustomOp& op);
  static u64 cumulative_size(const std::vector<CustomOp>& ops);
  std::string to_string() const;

  static void validate(const std::vector<CustomOp>& operations);

  friend std::ostream& operator<<(std::ostream& os, const CustomOp& op);
  bool operator==(const CustomOp& rhs) const;
  bool operator!=(const CustomOp& rhs) const;
};

struct MemoryRegionDefinition {
  /** Specifies the set of memory NUMA nodes on which benchmark data is to be allocated. If multiple nodes are set and
   * percentage_pages_first_partition is not set, pages are allocated in a round robin fashion. If
   * percentage_pages_first_partition is set, only two nodes are supported. percentage_pages_first_partition then
   * determines the share of the memory region located on the first node where the remaining part will be located on the
   * second node.
   */
  NumaNodeIDs node_ids = {};

  /** Specifies the number of NUMA nodes for the first partition of the NUMA node list. Ignored if not set. */
  std::optional<u64> node_count_first_partition = std::nullopt;

  /** Specifies the share of pages in percentage located in the first partition. Ignored if not set. */
  std::optional<u64> percentage_pages_first_partition = std::nullopt;

  /** Represents the total primary memory range to use for the benchmark. Must be a multiple of `access_size`.  */
  u64 size = 10 * GiB;

  /** Sepecify the use of huge pages in combination with `explicit_hugepages_size`. */
  bool transparent_huge_pages = false;

  /** Specify the used huge page size. Relevant when the OS supports multiple huge page sizes. Requires
   * `transparent_huge_pages` being set to true. When set to 0 while `transparent_huge_pages` is true, transparent huge
   * pages is enabled via madvise. */
  u64 explicit_hugepages_size = 0;

  PagePlacementMode placement_mode() const;
};

using MemoryRegionDefinitions = std::array<MemoryRegionDefinition, 2>;

/**
 * The values shown here define the benchmark and represent user-facing configuration options.
 */
struct BenchmarkConfig {
  /** Represent the memory region that one workload can use. Currently limited to two.*/
  MemoryRegionDefinitions memory_regions = {MemoryRegionDefinition{.size = 10 * GiB},
                                            MemoryRegionDefinition{.size = 0}};

  /** Represents the size of an individual memory access in Byte. Must be a power of two. */
  u32 access_size = 256;

  /** Represents the number of random access / custom operations to perform. Can *not* be set for sequential access. */
  u64 number_operations = 100'000'000;

  /** Number of threads to run the benchmark with. Must be a power of two. */
  u16 number_threads = 1;

  /** Alternative measure to end a benchmark by letting is run for `run_time` seconds. */
  u64 run_time = 0;

  /** Type of memory access operation to perform, i.e., read or write. */
  Operation operation = Operation::Read;

  /** Mode of execution, i.e., sequential, random, or custom. See `Mode` for all options. */
  Mode exec_mode = Mode::Sequential;

  /** Flush instruction to use after write operations. Only works with `Operation::Write`. See
   * `CacheInstruction` for more details on available options. */
  CacheInstruction cache_instruction = CacheInstruction::None;

  /** Specifies the set of NUMA nodes on which the benchmark threads are to run. */
  NumaNodeIDs numa_thread_nodes;

  ThreadPinMode thread_pin_mode = ThreadPinMode::AllNumaCores;

  /** Speficies the set of cores that the threads are pinned to. Only relevant for ThreadPinMode::SingleCoreFixed */
  CoreIDs thread_core_ids = {};

  /** Distribution to use for `Mode::Random`, i.e., uniform of zipfian. */
  RandomDistribution random_distribution = RandomDistribution::Uniform;

  /** Zipf skew factor for `Mode::Random` and `RandomDistribution::Zipf`. */
  double zipf_alpha = 0.9;

  /** List of custom operations to use in `Mode::Custom`. See `CustomOp` for more details on string representation.  */
  std::vector<CustomOp> custom_operations;

  /** Frequency in which to sample latency of custom operations. Only works in combination with `Mode::Custom`. */
  u64 latency_sample_frequency = 0;

  /** Represents the minimum size of an atomic work package. A batch contains batch_size / access_size number of
   * operations. The default value is 64 MiB (67108864B), a ~60 ms execution unit assuming the lowest bandwidth of
   * 1 GiB/s operations per thread. For dependent reads, this batch size determines the range in which addresses to
   * be accessed are randomly generated. For example, with a memory region size of 1 GB, a batch size of 64 MiB, and an
   * access size of 64 B, the first 1,048,576 (= 64 MiB / 64 B) access addresses point to a memory region of 64 MiB.
   * Narrowing the memory access range reduces TLB misses. */
  u64 min_io_batch_size = 64 * MiB;

  std::vector<std::string> matrix_args{};

  static BenchmarkConfig decode(YAML::Node& raw_config_data);
  void validate() const;
  bool contains_read_op() const;
  bool contains_write_op() const;
  bool contains_secondary_memory_op() const;

  std::string to_string(const std::string sep = ", ") const;
  nlohmann::json as_json() const;
};

struct ConfigEnums {
  // TODO(MW) use magic enum
  static const std::unordered_map<std::string, bool> str_to_mem_type;
  static const std::unordered_map<std::string, Mode> str_to_mode;
  static const std::unordered_map<std::string, Operation> str_to_operation;
  static const std::unordered_map<std::string, CacheInstruction> str_to_cache_instruction;
  static const std::unordered_map<std::string, RandomDistribution> str_to_random_distribution;
  static const std::unordered_map<std::string, ThreadPinMode> str_to_thread_pin_mode;

  // Map to convert a K/M/G suffix to the correct kibi, mebi-, gibibyte value.
  static const std::unordered_map<char, u64> scale_suffix_to_factor;
};

}  // namespace mema
