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
    set_memory_numa_nodes(data, memory_region_size, {node_id});
    utils::populate_memory(data, memory_region_size);

    for (size_t page_idx = 0; page_idx < region_page_count; ++page_idx) {
      auto addr = data + page_idx * utils::PAGE_SIZE;
      ASSERT_EQ(get_numa_node_index_by_address(addr), node_id);
    }
  }
}

}  // namespace mema
