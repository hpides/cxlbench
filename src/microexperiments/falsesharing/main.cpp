// Code based is based on: Ulrich Drepper. What Every Programmer Should Know About Memory. 2007.

#include <error.h>
#include <pthread.h>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "json.hpp"

// NOLINTBEGIN
#define MemaAssert(expr, msg)        \
  if (!static_cast<bool>(expr)) {    \
    std::cerr << (msg) << std::endl; \
  }                                  \
  static_assert(true, "End call of macro with a semicolon")
// NOLINTEND

// -- Types ------------------------------------------------------------------------------------------------------------
using u8 = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;
using u64 = uint64_t;
using i8 = int8_t;
using i16 = int16_t;
using i32 = int32_t;
using i64 = int64_t;
// Use Clock::now().
using Clock = std::chrono::steady_clock;
using DurationNs = u64;

struct Config {
  u32 thread_count;
  u32 size_factor_per_thread;
  u64 op_count;
  std::vector<DurationNs> durations;
};
// -- Constants --------------------------------------------------------------------------------------------------------
constexpr auto CACHELINE_SIZE = u64{64};
constexpr auto OP_COUNT = u64{10000000};
constexpr auto ITER_COUNT = u64{6};
// -- Utils ------------------------------------------------------------------------------------------------------------
DurationNs duration_in_ns(const auto& start, const auto& end) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

void to_json(nlohmann::json& json, const Config& config) {
  json["thread_count"] = config.thread_count;
  json["size_factor_per_thread"] = config.size_factor_per_thread;
  json["op_count"] = config.op_count;
  json["durations"] = config.durations;
}
// -- Benchmark --------------------------------------------------------------------------------------------------------
std::vector<DurationNs> run_benchmark(u32 thread_count, u32 size_factor_per_thread) {
  std::cout << "-- Run: #thread: " << thread_count << ", size factor: " << size_factor_per_thread << std::endl;
  const u32 value_count = thread_count * size_factor_per_thread > 0 ? thread_count * size_factor_per_thread : 1;
  u32 buffer_size = (value_count * sizeof(i64) + sizeof(std::vector<std::atomic<i64>>));
  std::cout << "value count: " << value_count << std::endl;

  // ensure size alignment
  if (buffer_size % CACHELINE_SIZE != 0) {
    buffer_size += (CACHELINE_SIZE - buffer_size % CACHELINE_SIZE);
  }
  std::cout << "buffer_size: " << buffer_size << std::endl;

  MemaAssert(buffer_size % CACHELINE_SIZE == 0, "buffer_size is not an integral multiple of CACHELINE_SIZE");
  auto* buffer = std::aligned_alloc(CACHELINE_SIZE, buffer_size);  // NOLINT
  MemaAssert(buffer != nullptr, "Memory allocation failed.");

  // Initialize atomic ints with 0.
  auto values = new (buffer) std::atomic<i64>[value_count];  // NOLINT
  // auto* values = new (buffer) std::vector<std::atomic<i64>>(value_count);

  // The main thread does not perform an operaiton. Instead, all the child thread (`threads`) perform the workload.
  const auto child_thread_count = thread_count - 1;
  auto threads = std::vector<std::thread>(child_thread_count);
  // -------------------------------------------------------------------------------------------------------------------
  auto perform_operations = [&values](auto position, auto /*thread_idx*/) {
    // perform atomic operations
    for (auto op_idx = u64{0}; op_idx < OP_COUNT; ++op_idx) {
      values[position].fetch_add(1);
      // std::cout << "T#" << thread_idx << " saw " << val << std::endl;
    }
  };
  // -------------------------------------------------------------------------------------------------------------------
  auto thread_task = [&](auto thread_idx) {
    // set affinity in thread attribute
    auto cpu_set = cpu_set_t{};
    CPU_ZERO(&cpu_set);
    CPU_SET(thread_idx, &cpu_set);
    const auto affinity_result = pthread_setaffinity_np(threads[thread_idx].native_handle(), sizeof(cpu_set), &cpu_set);
    if (affinity_result != 0) {
      std::cerr << "pthread_attr_setaffinity_np failed (error code: " + std::to_string(affinity_result) + ")."
                << std::endl;
      std::exit(1);
    }

    perform_operations(thread_idx * size_factor_per_thread, thread_idx);
  };
  // -------------------------------------------------------------------------------------------------------------------
  auto durations_ns = std::vector<DurationNs>(ITER_COUNT);
  // Run benchmark iterations.
  for (auto iteration_idx = u64{0}; iteration_idx <= ITER_COUNT; ++iteration_idx) {
    std::fill_n(values, value_count, 0);
    const auto start = Clock::now();

    // Create threads and start workload
    for (auto thread_idx = u64{0}; thread_idx < child_thread_count; ++thread_idx) {
      threads[thread_idx] = std::thread(thread_task, thread_idx);
    }

    // set affinity for main thread
    auto cpu_set = cpu_set_t{};
    CPU_ZERO(&cpu_set);
    CPU_SET(child_thread_count, &cpu_set);
    const auto affinity_result = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set), &cpu_set);
    if (affinity_result != 0) {
      std::cerr << "pthread_attr_setaffinity_np failed (error code: " + std::to_string(affinity_result) + ")."
                << std::endl;
      std::exit(1);
    }

    // perform atomic operations
    perform_operations((child_thread_count)*size_factor_per_thread, child_thread_count);

    // Join threads
    for (auto& thread : threads) {
      thread.join();
    }
    const auto end = Clock::now();
    durations_ns[iteration_idx] = duration_in_ns(start, end);

    // Verify atomic counters.
    if ((size_factor_per_thread == 0 && values[0].load() != thread_count * OP_COUNT) ||
        (size_factor_per_thread != 0 && values[0].load() != OP_COUNT)) {
      std::cerr << "values[0] wrong: " << values[0].load() << " instead of "
                << (size_factor_per_thread == 0 ? thread_count * OP_COUNT : OP_COUNT) << std::endl;
      std::exit(1);
    }

    for (auto thread_idx = u64{1}; thread_idx < thread_count; ++thread_idx) {
      if (size_factor_per_thread != 0 && values[thread_idx * size_factor_per_thread].load() != OP_COUNT) {
        std::cerr << "values[" << thread_idx << "] wrong: " << values[thread_idx * size_factor_per_thread].load()
                  << " instead of " << OP_COUNT;
        std::exit(1);
      }
    }
  }
  free(buffer);  // NOLINT
  return durations_ns;
}
// -- main -------------------------------------------------------------------------------------------------------------
int main(int argc, char* argv[]) {
  std::cout << "op count: " << OP_COUNT << std::endl;
  std::cout << "iter count: " << ITER_COUNT << std::endl;

  // Prepare result
  auto thread_counts = std::vector<u32>{2, 4, 8, 16};
  auto size_factors_per_thread = std::vector<u32>{0, 8};
  auto configs = std::vector<Config>{};
  configs.reserve(thread_counts.size() * size_factors_per_thread.size());

  for (auto& thread_count : thread_counts) {
    for (auto& size_factor : size_factors_per_thread) {
      configs.push_back({thread_count, size_factor, OP_COUNT, {}});
    }
  }

  // Run benchmarks
  for (auto& config : configs) {
    config.durations = run_benchmark(config.thread_count, config.size_factor_per_thread);
  }

  // Result output
  for (auto& config : configs) {
    std::cout << "-- #threads: " << config.thread_count << ", size factor: " << config.size_factor_per_thread
              << std::endl;
    std::cout << "durations (ns): " << std::endl;
    for (auto& duration : config.durations) {
      std::cout << "  " << duration << std::endl;
    }
  }

  auto json = nlohmann::json{};
  json["results"] = configs;
  auto file_stream = std::ofstream("false-sharing.json");
  file_stream << json;
  return 0;
}
