#pragma once

#include <numaif.h>

#include <filesystem>
#include <vector>

#include "benchmark.hpp"
#include "utils.hpp"

namespace mema {

// Different modes are possible, e.g., MPOL_BIND (no interleaving), MPOL_INTERLEAVE (interleaving, opt. bandwidth)
static auto MBIND_MODE = MPOL_INTERLEAVE;
static auto MBIND_FLAGS = MPOL_MF_STRICT | MPOL_MF_MOVE;

// Sets the numa policy to strict if numa is available and returns the number of available numa nodes.
uint8_t init_numa();

void set_task_numa_nodes(const NumaNodeIDs& node_ids);

void set_memory_numa_nodes(void* addr, const size_t memory_size, const NumaNodeIDs& node_ids);

void log_numa_nodes(spdlog::level::level_enum log_level, const std::string& message, const NumaNodeIDs& nodes);

void log_permissions_for_numa_nodes(spdlog::level::level_enum log_level, uint64_t thread_id);

void log_permissions_for_numa_nodes(spdlog::level::level_enum log_level, std::string thread_description);

// Returns node IDs on which the current thread is allowed to run on.
NumaNodeIDs get_numa_task_nodes();

// Returns the NumaNodeID of the node on which the memory for a given address is currently allocated. Mind that it's
// about an actual node, not the nodes on which the memory is allowed to be allocated.
NumaNodeID get_numa_node_index_by_address(char* const addr);

}  // namespace mema
