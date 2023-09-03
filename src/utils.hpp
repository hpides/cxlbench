#pragma once

#include <asm-generic/mman-common.h>
#include <asm-generic/mman.h>
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

// Maps an anonymous memory region. No data is mapped if `expected_length` is 0.
char* map(const size_t expected_length, const bool use_huge_pages, const NumaNodeIDs& numa_memory_nodes);

NumaNodeID get_numa_task_node();

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
  throw std::invalid_argument("Unknown enum value for " + std::string(typeid(T).name()));
}

}  // namespace utils
}  // namespace mema
