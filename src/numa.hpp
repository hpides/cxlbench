#pragma once

#include <filesystem>
#include <vector>

#include "benchmark.hpp"
#include "utils.hpp"

namespace mema {

void log_numa_nodes(const std::vector<uint64_t>& nodes);

void set_task_on_numa_nodes(const NumaNodeIDs& node_ids, const size_t num_numa_nodes);

void init_numa(const NumaNodeIDs& numa_nodes);

void set_memory_on_numa_nodes(void* addr, const size_t memory_size, const NumaNodeIDs& node_ids);

}  // namespace mema
