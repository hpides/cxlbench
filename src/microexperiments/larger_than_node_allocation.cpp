#include <numa.h>
#include <numaif.h>
#include <sys/mman.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include "benchmark/benchmark.h"

namespace {

#define Assert(expr, msg)         \
  if (!static_cast<bool>(expr)) { \
    throw std::logic_error{msg};  \
  }                               \
  static_assert(true, "End call of macro with a semicolon")

constexpr auto KiB = uint64_t{1024u};
constexpr auto MiB = 1024 * KiB;
constexpr auto GiB = 1024 * MiB;

}  // namespace

int main(int argc, char** argv) {
  if (argc != 4) {
    throw std::invalid_argument{"Need to specify <GiB size to allocate> <thread count> <use mlock (1 or 0)> "};
  }

  const auto thread_count = std::stoi(argv[2]);
  Assert(thread_count > 0, "Thread count must be greater than 0.");

  auto threads = std::vector<std::thread>{};
  threads.reserve(thread_count);

  const auto use_mlock = std::stoi(argv[3]);
  Assert(use_mlock == 0 || use_mlock == 1, "Use mlock must be 0 or 1.");

  if (use_mlock == 1) {
    std::cout << "Locking memory..." << std::endl;
    const auto ret = mlockall(MCL_CURRENT | MCL_FUTURE);
    Assert(ret == 0, "Failed to lock memory.");
  }

  const auto size_in_gb = std::stoi(argv[1]);
  Assert(size_in_gb > 0, "Size must be greater than 0.");

  const auto value_count = ((static_cast<uint64_t>(size_in_gb) * GiB) / sizeof(uint64_t)) + 1;
  const auto value_count_per_thread = (value_count / thread_count) + 1;
  std::cout << "Total required uint64_t values: " << value_count << std::endl;
  std::cout << "Total required uint64_t values per thread: " << value_count_per_thread << std::endl;

  auto values = std::make_shared<std::vector<uint64_t>>();
  values->resize(value_count_per_thread * thread_count);

  auto write_data = [&](uint16_t thread_idx) {
    std::cout << "Writing " << value_count_per_thread << " values..." << std::endl;

    auto last_node = int32_t{-1};
    const auto start_offset = thread_idx * value_count_per_thread;
    for (auto value_idx = uint64_t{0}; value_idx < value_count_per_thread; ++value_idx) {
      auto& value = (*values)[start_offset + value_idx];
      value = value_idx;

      if (!(value_idx % 50000000)) {
        auto identified_node_idx = int32_t{};
        auto value_ptr = reinterpret_cast<void*>(&value);

        const auto ret = move_pages(0, 1, &value_ptr, NULL, &identified_node_idx, 0);
        Assert(ret == 0, "Failed to determine the NUMA node for a given address.");

        std::cout << "Thread " << thread_idx << ": " << value_idx << " written on node " << identified_node_idx
                  << std::endl;

        if (last_node != identified_node_idx && last_node != -1) {
          break;
        }

        last_node = identified_node_idx;
      }
    }
    benchmark::DoNotOptimize(values);
  };

  for (auto i = uint16_t{0}; i < thread_count; ++i) {
    threads.emplace_back(write_data, i);
  }

  // wait for all threads
  for (std::thread& thread : threads) {
    thread.join();
  }
}
