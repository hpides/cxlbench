#pragma once

#include <array>
#include <optional>
#include <ostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "json.hpp"
#include "yaml-cpp/yaml.h"

namespace mema {

using u8 = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;
using u64 = uint64_t;
using i8 = int8_t;
using i16 = int16_t;
using i32 = int32_t;
using i64 = int64_t;

using NumaNodeID = uint16_t;
using NumaNodeIDs = std::vector<NumaNodeID>;
using InterleavingWeights = std::vector<uint32_t>;
using MemoryRegions = std::vector<char*>;
using CoreID = uint64_t;
using CoreIDs = std::vector<CoreID>;

enum class Mode : uint8_t { Sequential, Sequential_Desc, Random, Custom, DependentReads };

enum class RandomDistribution : uint8_t { Uniform, Zipf };

enum class FlushInstruction : uint8_t { Cache, NoCache, None };

enum class Operation : uint8_t { Read, Write };

enum class MemoryType : uint8_t { Primary, Secondary };

enum class PagePlacementMode : uint8_t { Interleaved, WeightedInterleaved, Partitioned };

// AllNumaCores pins a thread to the set of cores of a given NUMA node.
// SingleNumaCoreIncrement pins each thread to a single core, which is in the set of cores of a given NUMA node. The
// tool automatically determines the next available core when configuring the threads.
// SingleCoreFixed pints each thread to a single core. This is fully user-defined. The cores to pin threads at need to
// match the thread count for the workload.
enum class ThreadPinMode : uint8_t { AllNumaCores, SingleNumaCoreIncrement, SingleCoreFixed };

static constexpr size_t KiB = 1024u;
static constexpr size_t MiB = KiB * 1024;
static constexpr size_t GiB = MiB * 1024;
static constexpr size_t SECONDS_IN_NANOSECONDS = 1e9;

constexpr auto MEM_REGION_COUNT = uint64_t{2};
constexpr auto PAR_WORKLOAD_COUNT = uint64_t{2};
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
 * For writes: w_<size>_<flush_instruction>(_<offset>)
 *
 * with:
 * 'w' for write,
 * <size> is the size of the access (must be power of 2),
 * <flush_instruction> is the instruction to use after the write (none, cache, noache),
 * (optional) <offset> is the offset to the previously accessed address (can be negative, default is 0)
 *
 * */
struct CustomOp {
  MemoryType memory_type = MemoryType::Primary;
  Operation type;
  uint64_t size;
  FlushInstruction flush = FlushInstruction::None;
  // This can be signed, e.g., to represent the case when the previous cache line should be written to.
  int64_t offset = 0;

  static CustomOp from_string(const std::string& str);
  static std::vector<CustomOp> all_from_string(const std::string& str);
  static std::string all_to_string(const std::vector<CustomOp>& ops);
  static std::string to_string(const CustomOp& op);
  static uint64_t cumulative_size(const std::vector<CustomOp>& ops);
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

  /** Specifies node weights for page interleaving. */
  InterleavingWeights node_weights = {};

  /** Specifies the number of NUMA nodes for the first partition of the NUMA node list. Ignored if not set. */
  std::optional<uint64_t> node_count_first_partition = std::nullopt;

  /** Specifies the share of pages in percentage located in the first partition. Ignored if not set. */
  std::optional<uint64_t> percentage_pages_first_partition = std::nullopt;

  /** Represents the total primary memory range to use for the benchmark. Must be a multiple of `access_size`.  */
  uint64_t size = 10 * GiB;

  /** Sepecify the use of huge pages in combination with `explicit_hugepages_size`. */
  bool transparent_huge_pages = false;

  /** Specify the used huge page size. Relevant when the OS supports multiple huge page sizes. Requires
   * `transparent_huge_pages` being set to true. When set to 0 while `transparent_huge_pages` is true, transparent huge
   * pages is enabled via madvise. */
  uint64_t explicit_hugepages_size = 0;

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
  uint32_t access_size = 256;

  /** Represents the number of random access / custom operations to perform. Can *not* be set for sequential access. */
  uint64_t number_operations = 100'000'000;

  /** Number of threads to run the benchmark with. Must be a power of two. */
  uint16_t number_threads = 1;

  /** Alternative measure to end a benchmark by letting is run for `run_time` seconds. */
  uint64_t run_time = 0;

  /** Type of memory access operation to perform, i.e., read or write. */
  Operation operation = Operation::Read;

  /** Mode of execution, i.e., sequential, random, or custom. See `Mode` for all options. */
  Mode exec_mode = Mode::Sequential;

  /** Flush instruction to use after write operations. Only works with `Operation::Write`. See
   * `FlushInstruction` for more details on available options. */
  FlushInstruction flush_instruction = FlushInstruction::None;

  /** Deprecated. Number of disjoint memory regions to partition the `memory_region_size` into. Must be 0 or a divisor
   * of `number_threads` i.e., one or more threads map to one partition. When set to 0, it is equal to the number of
   * threads, i.e., each thread has its own partition. Default is set to 1.  */
  uint16_t number_partitions = 1;

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
  uint64_t latency_sample_frequency = 0;

  /** Represents the minimum size of an atomic work package. A chunk contains chunk_size / access_size number of
   * operations. The default value is 64 MiB (67108864B), a ~60 ms execution unit assuming the lowest bandwidth of
   * 1 GiB/s operations per thread. For dependent reads, this chunk size determines the range in which addresses to
   * be accessed are randomly generated. For example, with a memory region size of 1 GB, a chunk size of 64 MiB, and an
   * access size of 64 B, the first 1,048,576 (= 64 MiB / 64 B) access addresses point to a memory region of 64 MiB.
   * Narrowing the memory access range reduces TLB misses. */
  uint64_t min_io_chunk_size = 64 * MiB;

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
  static const std::unordered_map<std::string, FlushInstruction> str_to_flush_instruction;
  static const std::unordered_map<std::string, RandomDistribution> str_to_random_distribution;
  static const std::unordered_map<std::string, ThreadPinMode> str_to_thread_pin_mode;

  // Map to convert a K/M/G suffix to the correct kibi, mebi-, gibibyte value.
  static const std::unordered_map<char, uint64_t> scale_suffix_to_factor;
};

}  // namespace mema
