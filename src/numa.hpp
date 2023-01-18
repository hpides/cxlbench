#pragma once

#include <filesystem>
#include <vector>

namespace perma {

void log_numa_nodes(const std::vector<uint64_t>& nodes);

std::vector<uint64_t> auto_detect_numa(const std::filesystem::path& pmem_dir, const size_t num_numa_nodes);

void set_numa_nodes(const std::vector<uint64_t>& nodes, const size_t num_numa_nodes);

void init_numa(const std::filesystem::path& pmem_dir, const std::vector<uint64_t>& arg_nodes, bool is_dram,
               bool ignore_numa);

// Returns the numa nodes that the current task is not allowed to run (`set nodes`) on and that have a numa distance
// larger than NUMA_FAR_DISTANCE to the `set nodes`.
std::vector<uint64_t> get_far_nodes();

void set_to_far_cpus();

bool has_far_numa_nodes();

}  // namespace perma
