#include "numa.hpp"

#include <numa.h>
#include <numaif.h>
#include <string.h>

#include "gtest/gtest.h"
#include "test_utils.hpp"
#include "utils.hpp"

namespace mema {

class NumaTest : public BaseTest {
 protected:
  void SetUp() override { init_numa(); }
};

TEST_F(NumaTest, RetrieveCorrectNumaTaskNode) {
  const auto numa_max_node_id = numa_max_node();
  auto* const allowed_run_node_mask = numa_get_run_node_mask();

  for (auto node_id = NumaNodeID{0}; node_id <= numa_max_node_id; ++node_id) {
    if (!numa_bitmask_isbitset(allowed_run_node_mask, node_id)) {
      continue;
    }

    set_task_numa_nodes(NumaNodeIDs{node_id});
    EXPECT_EQ(get_numa_task_nodes(), NumaNodeIDs{node_id});
    numa_run_on_node_mask(allowed_run_node_mask);
  }
}

TEST_F(NumaTest, MemoryAllocationOnNode) {
  const auto numa_max_node_id = numa_max_node();
  auto* const allowed_memory_nodes_mask = numa_get_mems_allowed();

  constexpr auto memory_region_size = 1 * GIB_IN_BYTES;
  MemaAssert(memory_region_size % utils::PAGE_SIZE == 0, "Memory region needs to be a multiple of the page size.");
  constexpr auto region_page_count = memory_region_size / utils::PAGE_SIZE;

  for (auto node_id = NumaNodeID{0}; node_id <= numa_max_node_id; ++node_id) {
    if (!numa_bitmask_isbitset(allowed_memory_nodes_mask, node_id)) {
      continue;
    }

    char* data = utils::map(memory_region_size, true, 0);
    bind_memory_interleaved(data, memory_region_size, {node_id});
    utils::populate_memory(data, memory_region_size);

    for (size_t page_idx = 0; page_idx < region_page_count; ++page_idx) {
      auto addr = data + page_idx * utils::PAGE_SIZE;
      ASSERT_EQ(get_numa_node_index_by_address(addr), node_id);
    }
  }
}

TEST_F(NumaTest, FillPageLocationsPartitioned) {
  auto page_locations = PageLocations{};
  constexpr auto memory_region_size = size_t{10 * 1024 * 1024};
  const auto target_nodes = NumaNodeIDs{0, 1};
  constexpr auto percentage_first_node = 25u;

  fill_page_locations_partitioned(page_locations, memory_region_size, target_nodes, percentage_first_node);

  const auto region_page_count = memory_region_size / utils::PAGE_SIZE;
  const auto expected_first_node_page_count =
      static_cast<uint32_t>((percentage_first_node / 100.f) * region_page_count);
  auto page_idx = uint32_t{0};
  for (; page_idx < expected_first_node_page_count; ++page_idx) {
    ASSERT_EQ(page_locations[page_idx], target_nodes[0]);
  }
  for (; page_idx < region_page_count; ++page_idx) {
    ASSERT_EQ(page_locations[page_idx], target_nodes[1]);
  }
}

TEST_F(NumaTest, FillPageLocationsPartitionedFailure) {
  auto page_locations = PageLocations{};
  auto memory_region_size = size_t{10 * 1024 * 1024};
  auto one_target_node = NumaNodeIDs{0};
  auto three_target_nodes = NumaNodeIDs{0, 1, 2};
  auto percentage_first_node = 25;

  // Expect throw since exactly two nodes are required.
  EXPECT_THROW(
      fill_page_locations_partitioned(page_locations, memory_region_size, one_target_node, percentage_first_node),
      MemaException);
  EXPECT_THROW(
      fill_page_locations_partitioned(page_locations, memory_region_size, three_target_nodes, percentage_first_node),
      MemaException);
}

TEST_F(NumaTest, FillPageLocationsRoundRobin) {
  auto page_locations = PageLocations{};
  constexpr auto memory_region_size = size_t{10 * 1024 * 1024};
  const auto region_page_count = memory_region_size / utils::PAGE_SIZE;
  auto target_nodes = NumaNodeIDs{};

  auto assert_page_locations = [&] {
    const auto node_count = target_nodes.size();
    for (auto page_idx = uint32_t{0}; page_idx < region_page_count; ++page_idx) {
      ASSERT_EQ(page_locations[page_idx], target_nodes[page_idx % node_count]);
    }
  };

  {
    target_nodes = NumaNodeIDs{0};
    fill_page_locations_round_robin(page_locations, memory_region_size, target_nodes);
    assert_page_locations();
  }

  {
    target_nodes = NumaNodeIDs{0, 1};
    fill_page_locations_round_robin(page_locations, memory_region_size, target_nodes);
    assert_page_locations();
  }

  {
    target_nodes = NumaNodeIDs{0, 1, 2};
    fill_page_locations_round_robin(page_locations, memory_region_size, target_nodes);
    assert_page_locations();
  }
}

}  // namespace mema
