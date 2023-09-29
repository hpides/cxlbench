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

namespace {
constexpr uint32_t MIB_IN_BYTES = 1024 * 1024;
}

namespace mema {

class NumaReadWriteTest : public BaseTest {
 protected:
  void SetUp() override { init_numa(); }

  NumaNodeID get_numa_node_index_by_address(char* addr) {
    auto node = int32_t{};

    auto addr_ptr = reinterpret_cast<void*>(addr);
    auto ret = move_pages(0, 1, &addr_ptr, NULL, &node, 0);

    if (ret != 0) {
      throw std::runtime_error("move_pages() failed to determine NUMA node for address.");
    }

    return static_cast<NumaNodeID>(node);
  }
};

TEST_F(NumaReadWriteTest, SimpleWriteRead) {
  const auto memory_region_size = 100 * MIB_IN_BYTES;
  const auto numa_node_count = numa_num_configured_nodes();
  for (auto numa_idx = NumaNodeID{0}; numa_idx < numa_node_count; ++numa_idx) {
    SCOPED_TRACE("NUMA node index: " + std::to_string(numa_idx));
    char* base_addr =
        static_cast<char*>(mmap(nullptr, memory_region_size, PROT_READ | PROT_WRITE, utils::MAP_FLAGS, -1, 0));
    ASSERT_NE(base_addr, MAP_FAILED);
    ASSERT_NE(base_addr, nullptr);
    set_memory_on_numa_nodes(base_addr, memory_region_size, {numa_idx});
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
      ASSERT_EQ(get_numa_node_index_by_address(addr), numa_idx);
    }

    munmap(base_addr, memory_region_size);
  }
}

TEST_F(NumaReadWriteTest, IntrinsicsWriteRead) {
  const auto memory_region_size = 100 * MIB_IN_BYTES;
  const auto numa_node_count = numa_num_configured_nodes();
  for (auto numa_idx = NumaNodeID{0}; numa_idx < numa_node_count; ++numa_idx) {
    SCOPED_TRACE("NUMA node index: " + std::to_string(numa_idx));
    char* base_addr =
        static_cast<char*>(mmap(nullptr, memory_region_size, PROT_READ | PROT_WRITE, utils::MAP_FLAGS, -1, 0));
    ASSERT_NE(base_addr, MAP_FAILED);
    ASSERT_NE(base_addr, nullptr);
    set_memory_on_numa_nodes(base_addr, memory_region_size, {numa_idx});
    const auto cache_line_count = memory_region_size / rw_ops::CACHE_LINE_SIZE;
    for (auto cache_line_idx = size_t{0}; cache_line_idx < cache_line_count; ++cache_line_idx) {
      const auto addr = base_addr + (cache_line_idx * rw_ops::CACHE_LINE_SIZE);

      // Verify that the data at the memory region is not equal to WRITE_DATA.
      char result_buffer[64] __attribute__((aligned(64))) = {};
      auto* vec64_result_buffer = reinterpret_cast<mema::rw_ops::CharVec64*>(result_buffer);
      vec64_result_buffer[0] = mema::rw_ops::read_64(addr);

      auto compare_result = std::memcmp(result_buffer, rw_ops::WRITE_DATA, 64);
      ASSERT_NE(compare_result, 0);

      // Write data to memory region.
      rw_ops::write_none_64(addr);

      // Verify that the data at the memory region is equal to WRITE_DATA.
      vec64_result_buffer[0] = mema::rw_ops::read_64(addr);

      compare_result = std::memcmp(result_buffer, rw_ops::WRITE_DATA, 64);
      ASSERT_EQ(compare_result, 0);

      // Check if data is allocated on the correct numa node.
      ASSERT_EQ(get_numa_node_index_by_address(addr), numa_idx);
    }

    munmap(base_addr, memory_region_size);
  }
}

}  // namespace mema
