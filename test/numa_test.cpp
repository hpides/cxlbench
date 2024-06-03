#include "numa.hpp"

#include <numa.h>
#include <numaif.h>
#include <string.h>

#include <iostream>

#include "gtest/gtest.h"
#include "test_utils.hpp"
#include "utils.hpp"

namespace mema {

class NumaTest : public BaseTest {
 protected:
  void SetUp() override {
    init_numa();

    const auto numa_max_node_id = numa_max_node();
    auto* const allowed_memory_nodes_mask = numa_get_mems_allowed();
    valid_node_ids.reserve(numa_max_node_id);

    for (auto node_id = NumaNodeID{0}; node_id <= numa_max_node_id; ++node_id) {
      if (!numa_bitmask_isbitset(allowed_memory_nodes_mask, node_id)) {
        continue;
      }
      valid_node_ids.push_back(node_id);
    }
    valid_node_ids.shrink_to_fit();
  }

  NumaNodeIDs valid_node_ids{};
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

  constexpr auto memory_region_size = 1 * GiB;
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

TEST_F(NumaTest, PlacePages) {
  if (valid_node_ids.size() < 2) {
    GTEST_SKIP() << "Skipping test: system has " << valid_node_ids.size() << " but test requires at least 2.";
  }

  constexpr auto memory_region_size = 1 * MiB;
  MemaAssert(memory_region_size % utils::PAGE_SIZE == 0, "Memory region needs to be a multiple of the page size.");
  char* data = utils::map(memory_region_size, true, 0);
  utils::populate_memory(data, memory_region_size);
  constexpr auto region_page_count = memory_region_size / utils::PAGE_SIZE;
  ASSERT_EQ(region_page_count, 256);

  // Prepare page pointers for move_pages.
  auto pages = std::vector<void*>{};
  pages.resize(region_page_count);

  for (auto page_idx = uint64_t{0}; page_idx < region_page_count; ++page_idx) {
    pages[page_idx] = reinterpret_cast<void*>(data + page_idx * utils::PAGE_SIZE);
  }

  // -------------------------------------------------------------------------------------------------------------------
  auto test_place_one_node = [&](auto target_node) {
    SCOPED_TRACE("Target node: " + std::to_string(target_node));
    const auto locations = PageLocations(region_page_count, target_node);
    place_pages(data, memory_region_size, locations);

    // Retrieve and verify page status
    auto page_status = std::vector<int>(region_page_count, std::numeric_limits<int>::max());
    const auto ret = move_pages(0, region_page_count, pages.data(), NULL, page_status.data(), MPOL_MF_MOVE);
    const auto move_pages_errno = errno;
    ASSERT_EQ(ret, 0);

    for (auto page_idx = uint64_t{0}; page_idx < region_page_count; ++page_idx) {
      SCOPED_TRACE("Page id: " + std::to_string(page_idx));
      ASSERT_EQ(page_status[page_idx], target_node);
    }
  };
  // -------------------------------------------------------------------------------------------------------------------

  // Case 1: All pages on first node
  test_place_one_node(valid_node_ids[0]);

  // Case 2: All pages on second node
  test_place_one_node(valid_node_ids[1]);

  // Case 3: 60 % on first node, 40 % on second node
  auto locations = PageLocations(region_page_count, valid_node_ids[0]);
  const auto first_idx_on_second_node = static_cast<uint64_t>(0.6 * region_page_count);
  ASSERT_EQ(first_idx_on_second_node, 153);
  for (auto page_idx = uint64_t{0}; page_idx < region_page_count; ++page_idx) {
    if (page_idx < first_idx_on_second_node) {
      continue;
    }
    locations[page_idx] = valid_node_ids[1];
  }
  place_pages(data, memory_region_size, locations);

  // Retrieve and verify page status
  auto page_status = std::vector<int>(region_page_count, std::numeric_limits<int>::max());
  const auto ret = move_pages(0, region_page_count, pages.data(), NULL, page_status.data(), MPOL_MF_MOVE);
  const auto move_pages_errno = errno;
  ASSERT_EQ(ret, 0);

  for (auto page_idx = uint64_t{0}; page_idx < region_page_count; ++page_idx) {
    SCOPED_TRACE("Page id: " + std::to_string(page_idx));
    if (page_idx < first_idx_on_second_node) {
      ASSERT_EQ(page_status[page_idx], valid_node_ids[0]);
      continue;
    }
    ASSERT_EQ(page_status[page_idx], valid_node_ids[1]);
  }
}

TEST_F(NumaTest, VerifyPagePlacement) {
  if (valid_node_ids.size() < 2) {
    GTEST_SKIP() << "Skipping test: system has " << valid_node_ids.size() << " but test requires at least 2.";
  }

  constexpr auto memory_region_size = 1 * MiB;
  MemaAssert(memory_region_size % utils::PAGE_SIZE == 0, "Memory region needs to be a multiple of the page size.");
  char* data = utils::map(memory_region_size, true, 0);
  utils::populate_memory(data, memory_region_size);
  constexpr auto region_page_count = memory_region_size / utils::PAGE_SIZE;

  const auto nodes_first_node = NumaNodeIDs{valid_node_ids[0]};
  const auto nodes_second_node = NumaNodeIDs{valid_node_ids[1]};
  const auto nodes_first_second = NumaNodeIDs{valid_node_ids[0], valid_node_ids[1]};
  // Case 1: All pages on first node.
  {
    auto expected_page_locations = PageLocations{};
    fill_page_locations_round_robin(expected_page_locations, memory_region_size, nodes_first_node);
    place_pages(data, memory_region_size, expected_page_locations);
    ASSERT_TRUE(verify_interleaved_page_placement(data, memory_region_size, nodes_first_node));
    ASSERT_FALSE(verify_interleaved_page_placement(data, memory_region_size, nodes_second_node));
  }

  // Case 2: All pages on second node.
  {
    auto expected_page_locations = PageLocations{};
    fill_page_locations_round_robin(expected_page_locations, memory_region_size, nodes_second_node);
    place_pages(data, memory_region_size, expected_page_locations);
    ASSERT_FALSE(verify_interleaved_page_placement(data, memory_region_size, nodes_first_node));
    ASSERT_TRUE(verify_interleaved_page_placement(data, memory_region_size, nodes_second_node));
  }

  // Case 3: Pages interleaved
  {
    auto expected_page_locations = PageLocations{};
    fill_page_locations_round_robin(expected_page_locations, memory_region_size, nodes_first_second);
    place_pages(data, memory_region_size, expected_page_locations);
    ASSERT_FALSE(verify_interleaved_page_placement(data, memory_region_size, nodes_first_node));
    ASSERT_FALSE(verify_interleaved_page_placement(data, memory_region_size, nodes_second_node));
    ASSERT_TRUE(verify_interleaved_page_placement(data, memory_region_size, nodes_first_second));
  }

  // Case 4: Partitioned region
  {
    const auto nodes_first_first = NumaNodeIDs{valid_node_ids[0], valid_node_ids[0]};
    const auto nodes_second_second = NumaNodeIDs{valid_node_ids[1], valid_node_ids[1]};
    const auto percentage_first_node = uint64_t{64};
    auto expected_page_locations = PageLocations{};
    fill_page_locations_partitioned(expected_page_locations, memory_region_size, nodes_first_second,
                                    percentage_first_node);
    place_pages(data, memory_region_size, expected_page_locations);
    ASSERT_TRUE(verify_partitioned_page_placement(data, memory_region_size, nodes_first_second, percentage_first_node));
    ASSERT_FALSE(verify_interleaved_page_placement(data, memory_region_size, nodes_first_second));
    ASSERT_FALSE(verify_partitioned_page_placement(data, memory_region_size, nodes_first_first, percentage_first_node));
    ASSERT_FALSE(
        verify_partitioned_page_placement(data, memory_region_size, nodes_second_second, percentage_first_node));
  }
}

}  // namespace mema
