#pragma once

#include <numaif.h>

#include <filesystem>
#include <vector>

#include "benchmark.hpp"
#include "utils.hpp"

namespace mema {

constexpr auto PAGE_ERROR_LIMIT = 0.005f;

using PageLocations = std::vector<int>;

// Sets the numa policy to strict if numa is available and returns the number of available numa nodes.
uint8_t init_numa();

void set_task_numa_nodes(const NumaNodeIDs& node_ids);

void bind_memory_interleaved(void* addr, const size_t memory_size, const NumaNodeIDs& node_ids);

// Set the memory allocation policy to default (i.e., local allocation).
void set_interleave_memory_policy(const NumaNodeIDs& node_ids);

// Set the memory allocation policy to interleaved (MPOL_INTERLEAVED).
void set_default_memory_policy();

void fill_page_locations_round_robin(PageLocations& page_locations, size_t memory_region_size,
                                     const NumaNodeIDs& target_nodes);

void fill_page_locations_partitioned(PageLocations& page_locations, size_t memory_region_size,
                                     const NumaNodeIDs& target_nodes, const int32_t percentage_first_node);

void place_pages(char* const start_addr, size_t memory_region_size, const PageLocations& target_page_locations);

// Verify placement of pages. Returns false if more than PAGE_ERROR_LIMIT of pages are incorrectly located, true
// otherwise.
bool verify_interleaved_page_placement(char* const start_addr, size_t memory_region_size,
                                       const NumaNodeIDs& target_nodes);

// Check page locations based on given expected page locations.
bool verify_partitioned_page_placement(char* const start_addr, size_t memory_region_size,
                                       const PageLocations& expected_page_locations);

// Check page locations. Based on the given target nodes and percentage, the expected locations are calulated first,
// before the current page locations are compared with the expected ones.
bool verify_partitioned_page_placement(char* const start_addr, size_t memory_region_size,
                                       const NumaNodeIDs& target_nodes, const int32_t percentage_first_node);

void log_numa_nodes(spdlog::level::level_enum log_level, const std::string& message, const NumaNodeIDs& nodes);

void log_permissions_for_numa_nodes(spdlog::level::level_enum log_level, uint64_t thread_id);

void log_permissions_for_numa_nodes(spdlog::level::level_enum log_level, std::string thread_description);

// Returns node IDs on which the current thread is allowed to run on.
NumaNodeIDs get_numa_task_nodes();

// Returns the NumaNodeID of the node on which the memory for a given address is currently allocated. Mind that it's
// about an actual node, not the nodes on which the memory is allowed to be allocated.
NumaNodeID get_numa_node_index_by_address(char* const addr);

PageLocations get_page_locations(char* const start_addr, size_t memory_region_size);

}  // namespace mema
