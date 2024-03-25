#include "numa.hpp"

#include <numa.h>
#include <numaif.h>
#include <spdlog/spdlog.h>

#include <cstring>
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

void set_interleave_memory_policy(const NumaNodeIDs& node_ids) {
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
    spdlog::debug("Set NUMA bitmask for node ID {}.", node_id);
    numa_bitmask_setbit(nodemask, node_id);
  }
  set_mempolicy(MPOL_INTERLEAVE, nodemask->maskp, nodemask->size + 1);
}

void set_default_memory_policy() { set_mempolicy(MPOL_DEFAULT, nullptr, 0); }

void bind_memory_interleaved(void* addr, const size_t memory_size, const NumaNodeIDs& node_ids) {
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
    spdlog::debug("Set NUMA bitmask for node ID {}.", node_id);
    numa_bitmask_setbit(nodemask, node_id);
  }

  // Note that "[i]f the MPOL_INTERLEAVE policy was specified, pages already residing on the specified nodes will not be
  // moved such that they are interleaved", see mbind(2) manual.
  MemaAssert(nodemask != nullptr, "When setting the memory nodes, node mask cannot be nullptr.");
  const auto mbind_succeeded =
      mbind(addr, memory_size, MPOL_INTERLEAVE, nodemask->maskp, nodemask->size + 1, MPOL_MF_MOVE) == 0;
  const auto mbind_errno = errno;
  if (!mbind_succeeded) {
    spdlog::critical(
        "mbind failed: {}. You might have run out of space on the node(s) or reached the maximum map "
        "limit (vm.max_map_count).",
        strerror(mbind_errno));
    utils::crash_exit();
  }
  numa_bitmask_free(nodemask);
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

void fill_page_locations_round_robin(PageLocations& page_locations, size_t memory_region_size,
                                     const NumaNodeIDs& target_nodes) {
  spdlog::debug("Start filling page locations (round robin).");
  if (target_nodes.empty()) {
    spdlog::warn("Skipped filling page locations: no target NUMA node IDs are given.");
    return;
  }

  // Sort target nodes
  auto sorted_target_nodes = target_nodes;
  std::sort(sorted_target_nodes.begin(), sorted_target_nodes.end());
  const auto node_count = sorted_target_nodes.size();

  // Generate array
  const auto region_page_count = memory_region_size / utils::PAGE_SIZE;
  spdlog::debug("Page count of memory region: {}.", region_page_count);
  page_locations.resize(region_page_count);

  for (auto page_idx = uint64_t{0}; page_idx < region_page_count; ++page_idx) {
    page_locations[page_idx] = sorted_target_nodes[page_idx % node_count];
  }
}

void fill_page_locations_partitioned(PageLocations& page_locations, size_t memory_region_size,
                                     const NumaNodeIDs& target_nodes, const int32_t percentage_first_node) {
  spdlog::debug("Start filling page locations (partitioned).");
  MemaAssert(target_nodes.size() == 2, "When using partitioned page placements, only two NUMA nodes are supported.");
  MemaAssert(percentage_first_node >= 0 && percentage_first_node <= 100,
             "Percentage of pages on first node needs to be in range [0,100].");
  MemaAssert(memory_region_size % utils::PAGE_SIZE == 0, "Memory region size needs to be a multiple of the page size.");
  // Calculate page counts
  const auto region_page_count = memory_region_size / utils::PAGE_SIZE;
  const auto first_node_page_count = static_cast<uint32_t>((percentage_first_node / 100.f) * region_page_count);
  spdlog::info("Expected page locations: {} on node {}, {} on node {}.", first_node_page_count, target_nodes[0],
               region_page_count - first_node_page_count, target_nodes[1]);
  // Fill locations
  page_locations.resize(region_page_count);
  auto page_idx = uint32_t{0};
  for (; page_idx < first_node_page_count; page_idx++) {
    page_locations[page_idx] = target_nodes[0];
  }
  for (; page_idx < region_page_count; page_idx++) {
    page_locations[page_idx] = target_nodes[1];
  }
}

void place_pages(char* const start_addr, size_t memory_region_size, const PageLocations& target_page_locations) {
  spdlog::info("Starting page placement via move_pages.");
  if (target_page_locations.empty()) {
    spdlog::warn("Skipped page placement: no target page locations are given.");
    return;
  }

  const auto region_page_count = memory_region_size / utils::PAGE_SIZE;
  MemaAssert(region_page_count == target_page_locations.size(),
             "Passed target page location has incorrect number of locations.");

  // move_pages requires a vector of void*, one void* per page.
  auto pages = std::vector<void*>{};
  pages.resize(region_page_count);

  for (auto page_idx = uint64_t{0}; page_idx < region_page_count; ++page_idx) {
    pages[page_idx] = reinterpret_cast<void*>(start_addr + page_idx * utils::PAGE_SIZE);
  }

  // move_pages, initialize with value that is not a valid NUMA node idx. We assume that max int is not a valid idx.
  auto page_status = std::vector<int>(region_page_count, std::numeric_limits<int>::max());

  const auto ret =
      move_pages(0, region_page_count, pages.data(), target_page_locations.data(), page_status.data(), MPOL_MF_MOVE);
  const auto move_pages_errno = errno;

  if (ret != 0) {
    spdlog::critical("move_pages failed: {}", strerror(move_pages_errno));
    utils::crash_exit();
  }
}

bool verify_interleaved_page_placement(char* const start_addr, size_t memory_region_size,
                                       const NumaNodeIDs& target_nodes) {
  spdlog::debug("Check page placement errors.");
  if (target_nodes.empty()) {
    spdlog::warn("Skipped page placement verification.");
    return true;
  }

  // Generate array
  const auto region_page_count = memory_region_size / utils::PAGE_SIZE;
  spdlog::debug("Page count of memory region: {}.", region_page_count);
  // move_pages requires a vector of void*, one void* per page.
  auto pages = std::vector<void*>{};
  pages.resize(region_page_count);

  for (auto page_idx = uint64_t{0}; page_idx < region_page_count; ++page_idx) {
    pages[page_idx] = reinterpret_cast<void*>(start_addr + page_idx * utils::PAGE_SIZE);
  }

  // move_pages
  auto page_status = std::vector<int>(region_page_count, std::numeric_limits<int>::max());

  const auto ret = move_pages(0, region_page_count, pages.data(), NULL, page_status.data(), MPOL_MF_MOVE);
  const auto move_pages_errno = errno;

  if (ret != 0) {
    spdlog::critical("move_pages() failed: {}", strerror(move_pages_errno));
    utils::crash_exit();
  }

  // === Verify ===
  // Sort target nodes
  auto sorted_target_nodes = target_nodes;
  std::sort(sorted_target_nodes.begin(), sorted_target_nodes.end());
  const auto node_count = sorted_target_nodes.size();

  auto incorrect_placement_count = uint32_t{0};
  // offset in sorted target nodes
  auto sorted_position_offset = -1;
  auto inital_incorrect_page_locations = uint32_t{0};
  for (auto page_idx = uint64_t{0}; page_idx < region_page_count; ++page_idx) {
    // initial offset, if page location is one of the target nodes.
    if (sorted_position_offset == -1) {
      const auto iter = std::find(sorted_target_nodes.begin(), sorted_target_nodes.end(), page_status[page_idx]);
      if (iter == sorted_target_nodes.end()) {
        spdlog::info("Page {} has status {}. This is not one of the target node IDs.", page_idx, page_status[page_idx]);
        inital_incorrect_page_locations++;

        // --- early out ---
        static constexpr auto early_out_threshold = uint32_t{10};
        if (inital_incorrect_page_locations >= early_out_threshold) {
          spdlog::warn("Stopped page location verification sinice initial {} pages are on incorrect nodes.",
                       early_out_threshold);
          return false;
        }
        // -----------------
        continue;
      }
      sorted_position_offset = std::distance(sorted_target_nodes.begin(), iter);
      continue;
    }

    sorted_position_offset = (sorted_position_offset + 1) % node_count;
    const auto expected_location = sorted_target_nodes[sorted_position_offset];
    // spdlog::info("Page {} located on node {}, expected on node {}.", page_idx, page_status[page_idx],
    //                 expected_location);
    if (page_status[page_idx] != expected_location) {
      ++incorrect_placement_count;
      spdlog::debug("Page {} located on NUMA node {} but expected on {}.", page_idx, page_status[page_idx],
                    expected_location);
    }
  }

  if (incorrect_placement_count > 0) {
    spdlog::info("Page placement verification: {}/{} pages ({}%) are incorrectly placed.", incorrect_placement_count,
                 region_page_count,
                 static_cast<uint32_t>(((1.f * incorrect_placement_count) / region_page_count) * 100));
  }

  return incorrect_placement_count <= PAGE_ERROR_LIMIT * region_page_count;
}

bool verify_partitioned_page_placement(char* const start_addr, size_t memory_region_size,
                                       const PageLocations& expected_page_locations) {
  spdlog::debug("Check page placement errors.");
  if (expected_page_locations.empty()) {
    spdlog::warn("Skipped page placement verification.");
    return true;
  }

  const auto region_page_count = memory_region_size / utils::PAGE_SIZE;

  // Calculate page pointers.
  auto pages = std::vector<void*>{};
  pages.resize(region_page_count);
  for (auto page_idx = uint64_t{0}; page_idx < region_page_count; ++page_idx) {
    pages[page_idx] = reinterpret_cast<void*>(start_addr + page_idx * utils::PAGE_SIZE);
  }

  // Prepare page status array.
  auto page_status = std::vector<int>(region_page_count, std::numeric_limits<int>::max());

  const auto ret = move_pages(0, region_page_count, pages.data(), NULL, page_status.data(), MPOL_MF_MOVE);
  const auto move_pages_errno = errno;

  if (ret != 0) {
    spdlog::critical("move_pages() failed: {}", strerror(move_pages_errno));
    utils::crash_exit();
  }

  // Verify locations.
  auto incorrect_placement_count = uint32_t{0};
  for (auto page_idx = uint32_t{0}; page_idx < region_page_count; ++page_idx) {
    if (page_status[page_idx] < 0) {
      spdlog::critical("Expected status is error status: {}, {}", page_status[page_idx], strerror(move_pages_errno));
      utils::crash_exit();
    }
    if (page_status[page_idx] != expected_page_locations[page_idx]) {
      spdlog::debug("Page {}: status {}, expected: {}.", page_idx, page_status[page_idx],
                    expected_page_locations[page_idx]);
      ++incorrect_placement_count;
    }
  }

  if (incorrect_placement_count > 0) {
    spdlog::info("Page placement verification: {}/{} pages ({}%) are incorrectly placed.", incorrect_placement_count,
                 region_page_count,
                 static_cast<uint32_t>(((1.f * incorrect_placement_count) / region_page_count) * 100));
  }

  return incorrect_placement_count <= PAGE_ERROR_LIMIT * region_page_count;
}

bool verify_partitioned_page_placement(char* const start_addr, size_t memory_region_size,
                                       const NumaNodeIDs& target_nodes, const int32_t percentage_first_node) {
  auto page_locations = PageLocations{};
  fill_page_locations_partitioned(page_locations, memory_region_size, target_nodes, percentage_first_node);
  return verify_partitioned_page_placement(start_addr, memory_region_size, page_locations);
}

}  // namespace mema
