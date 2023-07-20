#include "numa.hpp"

#include <numa.h>
#include <numaif.h>
#include <spdlog/spdlog.h>

#include <sstream>

#include "benchmark.hpp"
#include "utils.hpp"

namespace mema {

void log_numa_nodes(const NumaNodeIDs& nodes) {
  const std::string used_nodes_str = std::accumulate(
      nodes.begin(), nodes.end(), std::string(),
      [](const auto& a, const auto b) -> std::string { return a + (a.length() > 0 ? ", " : "") + std::to_string(b); });
  spdlog::debug("Setting NUMA-affinity to node{}: {}", nodes.size() > 1 ? "s" : "", used_nodes_str);
}

void set_task_on_numa_nodes(const NumaNodeIDs& node_ids) {
  const size_t numa_node_count = numa_num_configured_nodes();

  if (node_ids.empty()) {
    spdlog::warn("No NUMA task nodes specified, task was not pinned to a NUMA node.");
    return;
  }

  // User specified numa nodes via the config file.
  if (numa_node_count < node_ids.size()) {
    spdlog::critical("More NUMA nodes specified than detected on server.");
    utils::crash_exit();
  }
  spdlog::debug("Setting NUMA task nodes according to config arguments.");
  log_numa_nodes(node_ids);

  if (numa_node_count < 2) {
    // Do nothing, as there isn't any affinity to be set.
    spdlog::debug("Not setting NUMA-awareness with {} node(s).", numa_node_count);
    return;
  }

  auto numa_nodemask = numa_bitmask_alloc(numa_node_count);
  for (const auto node_id : node_ids) {
    if (node_id >= numa_node_count) {
      spdlog::critical("Given numa task node id too large! (given: {}, max: {})", node_id, numa_node_count - 1);
      utils::crash_exit();
    }
    numa_bitmask_setbit(numa_nodemask, node_id);
  }

  // Perform task pinning.
  const auto success = numa_run_on_node_mask(numa_nodemask);

  if (success == -1) {
    spdlog::critical("Something went wrong while while pinning the task to a NUMA node.");
    utils::crash_exit();
  }

  // Mask deallocation.
  numa_free_nodemask(numa_nodemask);
}

void set_memory_on_numa_nodes(void* addr, const size_t memory_size, const NumaNodeIDs& node_ids) {
  if (node_ids.empty()) {
    spdlog::critical("Cannot set memory on numa nodes with an empty set of nodes.");
  }
  const auto num_memory_nodes = numa_num_configured_nodes();
  auto numa_nodemask = numa_bitmask_alloc(num_memory_nodes);
  auto numa_nodes_ss = std::stringstream{};
  for (const auto node_id : node_ids) {
    if (node_id >= num_memory_nodes) {
      spdlog::critical("Given numa memory node id too large! (given: {}, max: {})", node_id, num_memory_nodes - 1);
      utils::crash_exit();
    }
    numa_bitmask_setbit(numa_nodemask, node_id);
    numa_nodes_ss << " " << node_id;
  }
  numa_tonodemask_memory(addr, memory_size, numa_nodemask);
  numa_free_nodemask(numa_nodemask);
  spdlog::debug("Bound memory region {} to memory NUMA nodes{}.", addr, numa_nodes_ss.str());
}

uint8_t init_numa() {
  if (numa_available() < 0) {
    throw std::runtime_error("NUMA supported but could not be found!");
  }

  // Use a strict numa policy. Fail if memory cannot be allocated on a target node.
  numa_set_strict(1);
  return numa_num_configured_nodes();
}

}  // namespace mema
