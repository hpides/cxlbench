#include "numa.hpp"

#include <spdlog/spdlog.h>

#include <sstream>

#include "benchmark.hpp"
#include "read_write_ops.hpp"
#include "utils.hpp"

#ifdef HAS_NUMA
#include <numa.h>
#include <numaif.h>
#endif

namespace mema {

void log_numa_nodes(const NumaNodeIDs& nodes) {
  const std::string used_nodes_str = std::accumulate(
      nodes.begin(), nodes.end(), std::string(),
      [](const auto& a, const auto b) -> std::string { return a + (a.length() > 0 ? ", " : "") + std::to_string(b); });
  spdlog::info("Setting NUMA-affinity to node{}: {}", nodes.size() > 1 ? "s" : "", used_nodes_str);
}

void set_task_on_numa_nodes(const NumaNodeIDs& node_ids, const size_t num_numa_nodes) {
#ifndef HAS_NUMA
  spdlog::critical("Cannot set numa nodes without NUMA support.");
  utils::crash_exit();
#else
  auto numa_nodemask = numa_bitmask_alloc(num_numa_nodes);
  for (const auto node_id : node_ids) {
    if (node_id >= num_numa_nodes) {
      spdlog::critical("Given numa node id too large! (given: {}, max: {})", node_id, num_numa_nodes - 1);
      utils::crash_exit();
    }
    numa_bitmask_setbit(numa_nodemask, node_id);
  }

  // Perform task pinning.
  numa_run_on_node_mask(numa_nodemask);
  // Mask deallocation.
  numa_free_nodemask(numa_nodemask);
#endif
}

void set_memory_on_numa_nodes(void* addr, const size_t memory_size, const NumaNodeIDs& node_ids) {
#ifndef HAS_NUMA
  spdlog::critical("Cannot set memory on numa nodes without NUMA support.");
  utils::crash_exit();
#else
  if (node_ids.empty()) {
    spdlog::critical("Cannot set memory on numa nodes with an empty set of nodes.");
  }
  const auto num_memory_nodes = numa_num_configured_nodes();
  auto numa_nodemask = numa_bitmask_alloc(num_memory_nodes);
  auto numa_nodes_ss = std::stringstream{};
  for (const auto node_id : node_ids) {
    if (node_id >= num_memory_nodes) {
      spdlog::critical("Given numa node id too large! (given: {}, max: {})", node_id, num_memory_nodes - 1);
      utils::crash_exit();
    }
    numa_bitmask_setbit(numa_nodemask, node_id);
    numa_nodes_ss << " " << node_id;
  }
  numa_tonodemask_memory(addr, memory_size, numa_nodemask);
  numa_free_nodemask(numa_nodemask);
  spdlog::debug("Bound memory region {} to memory NUMA nodes{}.", addr, numa_nodes_ss.str());
#endif
}

void init_numa(const NumaNodeIDs& numa_nodes) {
#ifndef HAS_NUMA
  if (!numa_nodes.empty()) {
    spdlog::critical("Cannot explicitly set numa nodes without NUMA support.");
    utils::crash_exit();
  }

  // Don't do anything, as we don't have NUMA support.
  spdlog::warn("Running without NUMA-awareness.");
  return;
#else
  if (numa_available() < 0) {
    throw std::runtime_error("NUMA supported but could not be found!");
  }

  // Use a strict numa policy. Fail if memory cannot be allocated on a target node.
  numa_set_strict(1);
  const size_t num_numa_nodes = numa_num_configured_nodes();
  spdlog::info("Number of NUMA nodes in system: {}", num_numa_nodes);

  if (!numa_nodes.empty()) {
    // User specified numa nodes via the command line.
    if (num_numa_nodes < numa_nodes.size()) {
      spdlog::critical("More NUMA nodes specified than detected on server.");
      utils::crash_exit();
    }
    spdlog::info("Setting NUMA nodes according to command line arguments.");
    log_numa_nodes(numa_nodes);
    return set_task_on_numa_nodes(numa_nodes, num_numa_nodes);
  }

  if (num_numa_nodes < 2) {
    // Do nothing, as there isn't any affinity to be set.
    spdlog::info("Not setting NUMA-awareness with {} node(s).", num_numa_nodes);
    return;
  }
#endif
  spdlog::info("Thread was not pinned to a NUMA node.");
}

}  // namespace mema
