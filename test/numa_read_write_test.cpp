#include <numa.h>
#include <numaif.h>
#include <string.h>
#include <sys/mman.h>

#include <cstddef>
#include <cstring>
#include <string>
#include <unordered_set>

#include "gtest/gtest.h"
#include "numa.hpp"
#include "read_write_ops.hpp"
#include "test_utils.hpp"
#include "utils.hpp"

namespace mema {

class NumaReadWriteTest : public BaseTest {};

TEST_F(NumaReadWriteTest, SimpleWriteRead) {
  const auto memory_region_size = 100 * MiB;
  const auto max_node_id = numa_max_node();
  const auto* const memory_nodes_mask = numa_get_mems_allowed();
  for (auto node_id = NumaNodeID{0}; node_id <= max_node_id; ++node_id) {
    SCOPED_TRACE("NUMA node index: " + std::to_string(node_id));
    if (!numa_bitmask_isbitset(memory_nodes_mask, node_id)) {
      continue;
    }

    // Prepare memory region.
    char* base_addr =
        static_cast<char*>(mmap(nullptr, memory_region_size, PROT_READ | PROT_WRITE, utils::MAP_FLAGS, -1, 0));
    ASSERT_NE(base_addr, MAP_FAILED);
    ASSERT_NE(base_addr, nullptr);
    bind_memory_interleaved(base_addr, memory_region_size, {node_id});
    utils::populate_memory(base_addr, memory_region_size);
    verify_interleaved_page_placement(base_addr, memory_region_size, {node_id});

    // Write and read data.
    const auto cache_line_count = memory_region_size / rw_ops::CACHE_LINE_SIZE;
    for (auto cache_line_idx = size_t{0}; cache_line_idx < cache_line_count; ++cache_line_idx) {
      const auto addr = base_addr + (cache_line_idx * rw_ops::CACHE_LINE_SIZE);

      // Verify that the data at the memory region is not equal to WRITE_DATA.
      auto compare_result = std::memcmp(addr, rw_ops::WRITE_DATA, rw_ops::CACHE_LINE_SIZE);
      ASSERT_NE(compare_result, 0);

      // Write data to memory region.
      std::memcpy(addr, rw_ops::WRITE_DATA, rw_ops::CACHE_LINE_SIZE);

      // Verify that the data at the memory region is equal to WRITE_DATA.
      compare_result = std::memcmp(addr, rw_ops::WRITE_DATA, rw_ops::CACHE_LINE_SIZE);
      ASSERT_EQ(compare_result, 0);

      // Check if data is allocated on the correct numa node.
      ASSERT_EQ(numa_node_index_by_address(addr), node_id);
    }

    munmap(base_addr, memory_region_size);
  }
}

TEST_F(NumaReadWriteTest, IntrinsicsWriteRead) {
  const auto memory_region_size = 100 * MiB;
  const auto max_node_id = numa_max_node();
  const auto* const memory_nodes_mask = numa_get_mems_allowed();
  for (auto node_id = NumaNodeID{0}; node_id <= max_node_id; ++node_id) {
    SCOPED_TRACE("NUMA node index: " + std::to_string(node_id));
    if (!numa_bitmask_isbitset(memory_nodes_mask, node_id)) {
      continue;
    }

    // Prepare memory region
    char* base_addr =
        static_cast<char*>(mmap(nullptr, memory_region_size, PROT_READ | PROT_WRITE, utils::MAP_FLAGS, -1, 0));
    ASSERT_NE(base_addr, MAP_FAILED);
    ASSERT_NE(base_addr, nullptr);
    bind_memory_interleaved(base_addr, memory_region_size, {node_id});
    utils::populate_memory(base_addr, memory_region_size);
    verify_interleaved_page_placement(base_addr, memory_region_size, {node_id});

    // Write and read data
    const auto cache_line_count = memory_region_size / rw_ops::CACHE_LINE_SIZE;
    for (auto cache_line_idx = size_t{0}; cache_line_idx < cache_line_count; ++cache_line_idx) {
      const auto addr = base_addr + (cache_line_idx * rw_ops::CACHE_LINE_SIZE);

      // Verify that the data at the memory region is not equal to WRITE_DATA.
      char result_buffer[64] __attribute__((aligned(64))) = {};
      auto* vec_result_buffer = reinterpret_cast<rw_ops::CharVec*>(result_buffer);
      vec_result_buffer[0] = rw_ops::read_64(addr);

      auto compare_result = std::memcmp(result_buffer, rw_ops::WRITE_DATA, rw_ops::VECTOR_SIZE);
      ASSERT_NE(compare_result, 0);

      // Write data to memory region.
      rw_ops::write_none_64(addr);

      // Verify that the data at the memory region is equal to WRITE_DATA.
      vec_result_buffer[0] = rw_ops::read_64(addr);

      compare_result = std::memcmp(result_buffer, rw_ops::WRITE_DATA, rw_ops::VECTOR_SIZE);
      ASSERT_EQ(compare_result, 0);

      // Check if data is allocated on the correct numa node.
      ASSERT_EQ(numa_node_index_by_address(addr), node_id);
    }

    munmap(base_addr, memory_region_size);
  }
}

}  // namespace mema
