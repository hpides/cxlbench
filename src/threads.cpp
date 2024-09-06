#include "threads.hpp"

#include <pthread.h>

#include "benchmark.hpp"
#include "utils.hpp"

namespace cxlbench {

CoreIDs allowed_thread_core_ids() {
  auto cpu_set = cpu_set_t{};
  CPU_ZERO(&cpu_set);
  const auto thread_id = pthread_self();
  pthread_getaffinity_np(thread_id, sizeof(cpu_set), &cpu_set);

  auto core_ids = CoreIDs{};
  core_ids.reserve(CPU_SETSIZE);
  for (auto core_id = CoreID{0}; core_id < CPU_SETSIZE; ++core_id) {
    if (CPU_ISSET(core_id, &cpu_set)) {
      core_ids.emplace_back(core_id);
    }
  }
  core_ids.shrink_to_fit();
  spdlog::debug("Thread {} is pinned to cores [{}]", thread_id, utils::numbers_to_string(core_ids));
  return core_ids;
}

void pin_thread_to_cores(const CoreIDs& core_ids) {
  if (core_ids.empty()) {
    spdlog::warn("No core IDs specified, cannot pin thread to cores.");
    return;
  }

  auto cpu_set = cpu_set_t{};
  CPU_ZERO(&cpu_set);
  for (auto& core_id : core_ids) {
    CPU_SET(core_id, &cpu_set);
  }

  const auto thread_id = pthread_self();
  const auto pinning_result = pthread_setaffinity_np(thread_id, sizeof(cpu_set_t), &cpu_set);
  if (pinning_result != 0) {
    spdlog::critical("Pinning thread {} to cores [{}] failed. Error: {}", thread_id, utils::numbers_to_string(core_ids),
                     std::strerror(pinning_result));
    utils::crash_exit();
  }
  spdlog::debug("Pinned thread {} to cores [{}]", thread_id, utils::numbers_to_string(core_ids));
}

}  // namespace cxlbench
