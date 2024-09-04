#include <numa.h>
#include <numaif.h>
#include <sys/mman.h>

#include <array>
#include <cassert>
#include <chrono>
#include <cstring>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "benchmark/benchmark.h"

namespace {

#define Assert(expr, msg)         \
  if (!static_cast<bool>(expr)) { \
    throw std::logic_error{msg};  \
  }                               \
  static_assert(true, "End call of macro with a semicolon")

using u8 = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;
using u64 = uint64_t;
using i8 = int8_t;
using i16 = int16_t;
using i32 = int32_t;
using i64 = int64_t;

constexpr auto KiB = u32{1024};
constexpr auto MiB = 1024 * KiB;

// Used to clear the caches. Let's assume 500 MiB is sufficient.
constexpr auto LLCache_SIZE = 500 * MiB;
constexpr auto VALUE_COUNT = LLCache_SIZE / sizeof(u64);

constexpr auto RUNS = u16{10};

// Access the first byte of all cache lines.
constexpr auto CACHE_LINE_SIZE = u32{64};

// Use Clock::now().
using Clock = std::chrono::steady_clock;

u64 duration_in_nanoseconds(const auto& start, const auto& end) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

std::vector<std::byte> generate_random_data(size_t length) {
  const auto chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
  const size_t max_index = strlen(chars) - 1;

  std::mt19937 rng{};
  std::uniform_int_distribution<u64> dist(0, max_index);
  auto randchar = [&] { return static_cast<std::byte>(chars[dist(rng)]); };

  std::vector<std::byte> data(length);
  std::generate_n(data.begin(), length, randchar);
  return data;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    throw std::invalid_argument{"Need to specify <numa node index>"};
  }

  const u32 numa_node_idx = std::stoi(argv[1]);
  std::cout << "Numa node index: " << numa_node_idx << std::endl;

  const auto numa_node_count = numa_num_configured_nodes();
  Assert(numa_node_count > numa_node_idx, "Local node idx has to be < numa node count.");

  const auto size = 1 * MiB;
  void* addr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  Assert(addr != MAP_FAILED, "Failed to allocate memory.");
  numa_tonode_memory(addr, size, numa_node_idx);

  const auto random_data = generate_random_data(size);

  std::memcpy(addr, random_data.data(), size);
  Assert(std::memcmp(random_data.data(), addr, size) == 0, "Copied data is not equal.");

  // Check if data is on correct node.
  {
    auto identified_node_idx = i32{};
    const auto ret = move_pages(0, 1, &addr, NULL, &identified_node_idx, 0);
    Assert(ret == 0, "Failed to determine the NUMA node for a given address.");
    Assert(numa_node_idx == identified_node_idx, "Local node idx and identified node idx are not eqal.");
  }

  // Clear all caches.
  std::cout << "Clearing the caches..." << std::endl;
  auto values = std::make_unique<std::array<u64, VALUE_COUNT>>();
  for (auto counter = u32{0}; auto& value : *values) {
    value = counter;
    ++counter;
  }
  benchmark::DoNotOptimize(values);

  const auto cache_line_count = size / CACHE_LINE_SIZE;

  auto res = std::byte{};
  // Local memory access.
  std::cout << "Benchmarkinkg local access..." << std::endl;
  auto access_durations = std::array<u64, RUNS>{};
  auto bytes = reinterpret_cast<std::byte*>(addr);

  for (auto run = u16{0}; run < RUNS; ++run) {
    const auto start = Clock::now();

    for (auto access_idx = u32{0}; access_idx < cache_line_count; ++access_idx) {
      const auto& res = bytes[access_idx * CACHE_LINE_SIZE];
      benchmark::DoNotOptimize(res);
    }

    const auto end = Clock::now();
    access_durations[run] = duration_in_nanoseconds(start, end);
  }

  std::cout << "Latency for accessing the first byte of " << cache_line_count << " cache lines in ns:" << std::endl;
  for (const auto& duration : access_durations) {
    std::cout << duration << std::endl;
  }
}
