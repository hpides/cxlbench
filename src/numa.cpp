#include "numa.hpp"

#include <numa.h>
#include <numaif.h>
#include <spdlog/spdlog.h>

#include <sstream>
#include <string>

#include "benchmark.hpp"
#include "utils.hpp"

namespace mema {

void log_numa_nodes(spdlog::level::level_enum log_level, const std::string& message, const NumaNodeIDs& nodes) {
  const auto used_nodes_str = std::accumulate(
      nodes.begin(), nodes.end(), std::string(),
      [](const auto& a, const auto b) -> std::string { return a + (a.length() > 0 ? ", " : "") + std::to_string(b); });
  spdlog::log(log_level, "{} NUMA node{}: {}", message, nodes.size() > 1 ? "s" : "", used_nodes_str);
}

void set_task_numa_nodes(const NumaNodeIDs& node_ids) {
  if (node_ids.empty()) {
    spdlog::warn("No NUMA task nodes specified, task was not pinned to NUMA nodes.");
    return;
  }

  const size_t numa_node_count = numa_num_configured_nodes();
  const size_t max_node_id = numa_max_node();

  const auto* const allowed_run_node_mask = numa_get_run_node_mask();
  auto* const nodemask = numa_allocate_nodemask();
  for (const auto node_id : node_ids) {
    if (!numa_bitmask_isbitset(allowed_run_node_mask, node_id)) {
      spdlog::critical("Thread is not allowed to run on given node id (given: {}).", node_id);
      utils::crash_exit();
    }
    numa_bitmask_setbit(nodemask, node_id);
  }

  const auto task_pinned = numa_run_on_node_mask(nodemask) == 0;
  const auto task_pinning_errno = errno;
  numa_bitmask_free(nodemask);
  if (!task_pinned) {
    spdlog::critical("Pinning the task to NUMA nodes failed. Error: {}", std::strerror(task_pinning_errno));
    utils::crash_exit();
  }

  log_numa_nodes(spdlog::level::debug, "Bound thread to", node_ids);
}

void set_memory_numa_nodes(void* addr, const size_t memory_size, const NumaNodeIDs& node_ids) {
  if (node_ids.empty()) {
    spdlog::critical("Cannot set memory on numa nodes with an empty set of nodes.");
  }
  const auto* const allowed_memory_nodes_mask = numa_get_mems_allowed();
  auto* const nodemask = numa_allocate_nodemask();
  for (const auto node_id : node_ids) {
    if (!numa_bitmask_isbitset(allowed_memory_nodes_mask, node_id)) {
      spdlog::critical("Memory allocation on numa node id not allowed (given: {}).", node_id);
      utils::crash_exit();
    }
    numa_bitmask_setbit(nodemask, node_id);
  }

  MemaAssert(nodemask != nullptr, "When setting the memory nodes, node mask cannot be nullptr.");
  const auto mbind_succeeded =
      mbind(addr, memory_size, MBIND_MODE, nodemask->maskp, nodemask->size + 1, MBIND_FLAGS) == 0;
  const auto mbind_errno = errno;
  numa_bitmask_free(nodemask);
  if (!mbind_succeeded) {
    spdlog::critical(
        "mbind failed: {}. You might have run out of space on the node(s) or reached the maximum map "
        "limit (vm.max_map_count).",
        strerror(mbind_errno));
    utils::crash_exit();
  }
  auto stream = std::stringstream{};
  stream << "Bound memory region " << addr << " to";
  log_numa_nodes(spdlog::level::debug, stream.str(), node_ids);
}

uint8_t init_numa() {
  if (numa_available() < 0) {
    spdlog::critical("NUMA supported but could not be found!");
    utils::crash_exit();
  }

  // Use a strict numa policy. Fail if memory cannot be allocated on a target node.
  numa_set_strict(1);
  log_permissions_for_numa_nodes(spdlog::level::info, "Main");

  return numa_num_configured_nodes();
}

void log_permissions_for_numa_nodes(spdlog::level::level_enum log_level, uint64_t thread_id) {
  log_permissions_for_numa_nodes(log_level, std::to_string(thread_id));
}

void log_permissions_for_numa_nodes(spdlog::level::level_enum log_level, const std::string thread_description) {
  const size_t max_node_id = numa_max_node();

  auto log_numa = [&](auto mask, std::string_view description) {
    auto allowed_ids_stream = std::stringstream{};
    auto allowed_delim = "";
    auto forbidden_ids_stream = std::stringstream{};
    auto forbidden_delim = "";
    for (auto node_id = size_t{0}; node_id <= max_node_id; ++node_id) {
      if (numa_bitmask_isbitset(mask, node_id)) {
        allowed_ids_stream << allowed_delim << node_id;
        allowed_delim = ", ";
      } else {
        forbidden_ids_stream << forbidden_delim << node_id;
        forbidden_delim = ", ";
      }
    }
    spdlog::log(log_level, "Thread {}: {}: allowed nodes [{}], forbidden nodes: [{}].", thread_description, description,
                allowed_ids_stream.str(), forbidden_ids_stream.str());
  };

  log_numa(numa_get_run_node_mask(), "task binding");
  log_numa(numa_get_mems_allowed(), "memory allocation");
}

NumaNodeIDs get_numa_task_nodes() {
  const auto max_node_id = numa_max_node();
  auto run_nodes = NumaNodeIDs{};

  // Check which NUMA node the calling thread is allowed to run on.
  const auto* const allowed_run_node_mask = numa_get_run_node_mask();

  for (auto node_id = NumaNodeID{0}; node_id <= max_node_id; ++node_id) {
    if (numa_bitmask_isbitset(allowed_run_node_mask, node_id)) {
      run_nodes.emplace_back(node_id);
    }
  }

  if (run_nodes.empty()) {
    spdlog::critical("Could not determine NUMA task nodes of calling thread.");
    utils::crash_exit();
  }

  return run_nodes;
}

NumaNodeID get_numa_node_index_by_address(char* const addr) {
  auto node = int32_t{};

  auto addr_ptr = reinterpret_cast<void*>(addr);
  const auto ret = move_pages(0, 1, &addr_ptr, NULL, &node, 0);

  if (ret != 0) {
    spdlog::critical("move_pages() failed to determine NUMA node for address.");
    utils::crash_exit();
  }

  return static_cast<NumaNodeID>(node);
}

}  // namespace mema
