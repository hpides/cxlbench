#pragma once

#include <asm-generic/mman-common.h>
#include <asm-generic/mman.h>
#include <spdlog/spdlog.h>
#include <sys/mman.h>

#include <filesystem>
#include <unordered_map>
#include <vector>

#include "benchmark_config.hpp"
#include "json.hpp"

namespace mema {

class MemaException : public std::exception {
 public:
  const char* what() const noexcept override { return "Execution failed. Check logs for more details."; }
};

namespace utils {

static constexpr auto DATA_GEN_THREAD_COUNT = size_t{8};  // Should be a power of two
static constexpr auto PAGE_SIZE = size_t{4 * 1024ul};     // 4 KiB page size
static constexpr auto ONE_GB = size_t{1024ul * 1024 * 1024};
static constexpr auto SHORT_STRING_SIZE = size_t{1};

static int MAP_FLAGS = MAP_PRIVATE | MAP_ANONYMOUS;
static int MAP_FLAGS_HUGETLB = MAP_FLAGS | MAP_HUGETLB;

static constexpr auto LAST_RESULTS_FILENAME = "last_results.json";

// Maps an anonymous memory region. No data is mapped if `expected_length` is 0.
char* map(const uint64_t expected_length, const bool use_transparent_huge_pages, const uint64_t explicit_hugepages_size,
          const NumaNodeIDs& numa_memory_nodes);

// Calculates and returns the mmap page size flag for the given page size. Similar to MAP_HUGE_2MB and MAP_HUGE_1GB but
// dynamic for every given page size.
int mmap_page_size_mask(const uint32_t page_size);

NumaNodeIDs get_numa_task_nodes();

void generate_read_data(char* addr, const uint64_t memory_size);

void prefault_memory(char* addr, const uint64_t memory_size, const uint64_t page_size);

uint64_t duration_to_nanoseconds(const std::chrono::steady_clock::duration duration);

// Returns a Zipf random variable
uint64_t zipf(const double alpha, const uint64_t n);
double rand_val();

void crash_exit();
void print_segfault_error();

std::string get_time_string();
std::filesystem::path create_result_file(const std::filesystem::path& result_dir,
                                         const std::filesystem::path& config_path);
void write_benchmark_results(const std::filesystem::path& result_path, const nlohmann::json& results);

// Returns the first std::string that is mapped to as a specified enum value. If short_result is set, only strings of
// size SHORT_STRING_SIZE are returned.
template <typename T>
std::string get_enum_as_string(const std::unordered_map<std::string, T>& enum_map, T value, bool short_result = false) {
  for (auto it = enum_map.cbegin(); it != enum_map.cend(); ++it) {
    if (it->second == value) {
      const auto is_short_char = it->first.size() == SHORT_STRING_SIZE;
      if (short_result && is_short_char) {
        return it->first;
      } else if (!(short_result || is_short_char)) {
        return it->first;
      }
    }
  }
  spdlog::critical("Unknown enum value for {}", typeid(T).name());
  crash_exit();

  // This is only here to silence the "control reaches end of non-void function"
  // compiler warning. They were observed on GCC 12 and Clang 15.
  return "";
}

}  // namespace utils
}  // namespace mema
