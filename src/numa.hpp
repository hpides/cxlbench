#pragma once

#include <filesystem>
#include <vector>

#include "benchmark.hpp"
#include "utils.hpp"

namespace mema {

void log_numa_nodes(const std::vector<uint64_t>& nodes);

void set_task_on_numa_nodes(const NumaNodeIDs& node_ids);

// Sets the numa policy to strict if numa is available and returns the number of available numa nodes.
uint8_t init_numa();

void set_memory_on_numa_nodes(void* addr, const size_t memory_size, const NumaNodeIDs& node_ids);

void log_permissions_for_numa_nodes(spdlog::level::level_enum log_level, uint64_t thread_id);

void log_permissions_for_numa_nodes(spdlog::level::level_enum log_level, std::string thread_description);

}  // namespace mema
