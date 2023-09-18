#include <immintrin.h>
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

#define READ_SIMD_256(mem_addr, offset) _mm256_load_si256(reinterpret_cast<const __m256i*>((mem_addr) + ((offset)*32)))

#define READ_SIMD_512(mem_addr, offset) \
  _mm512_load_si512(reinterpret_cast<const void*>((mem_addr) + ((offset)*mema::rw_ops::CACHE_LINE_SIZE)))

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
    auto addresses = std::unordered_set<char*>{};
    for (auto cache_line_idx = size_t{0}; cache_line_idx < cache_line_count; ++cache_line_idx) {
      const auto addr = base_addr + (cache_line_idx * rw_ops::CACHE_LINE_SIZE);
      std::memcpy(addr, rw_ops::WRITE_DATA, rw_ops::CACHE_LINE_SIZE);
      const auto compare_result = std::memcmp(addr, rw_ops::WRITE_DATA, rw_ops::CACHE_LINE_SIZE);
      // check if all compared bytes were equal.
      ASSERT_EQ(compare_result, 0);
      ASSERT_FALSE((addresses.contains(addr)));
      addresses.insert(addr);
      ASSERT_TRUE((addresses.contains(addr)));
      // Check if data is allocated on the correct numa node.
      ASSERT_EQ(get_numa_node_index_by_address(addr), numa_idx);
    }

    munmap(base_addr, memory_region_size);
  }
}

#ifdef HAS_AVX_2
TEST_F(NumaReadWriteTest, AVX2WriteRead) {
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
      // write data to memory region via AVX2 intrinsics
      rw_ops::simd_write_none_64(addr);
      // read data from memory region via AVX2 intrinsics into SIMD registers
      __m256i read_result[2];
      read_result[0] = READ_SIMD_256(addr, 0);
      read_result[1] = READ_SIMD_256(addr, 1);
      // store data from SIMD registers into local char array
      char read_cache_line[rw_ops::CACHE_LINE_SIZE] __attribute__((aligned(64))) = {};
      _mm256_store_si256(reinterpret_cast<__m256i*>(read_cache_line), read_result[0]);
      _mm256_store_si256(reinterpret_cast<__m256i*>(read_cache_line + 32), read_result[1]);
      const auto compare_result = std::memcmp(read_cache_line, rw_ops::WRITE_DATA, rw_ops::CACHE_LINE_SIZE);
      // check if all compared bytes were equal.
      ASSERT_EQ(compare_result, 0);
      // Check if data is allocated on the correct numa node.
      ASSERT_EQ(get_numa_node_index_by_address(addr), numa_idx);
    }

    munmap(base_addr, memory_region_size);
  }
}
#endif

#ifdef HAS_AVX_512
TEST_F(NumaReadWriteTest, AVX512WriteRead) {
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
      // write data to memory region via AVX512 intrinsics
      rw_ops::simd_write_none_64(addr);
      // read data from memory region via AVX512 intrinsics into SIMD registers
      const __m512i read_result = READ_SIMD_512(addr, 0);
      // store data from SIMD registers into local char array
      char read_cache_line[rw_ops::CACHE_LINE_SIZE] __attribute__((aligned(64))) = {};
      _mm512_store_si512(read_cache_line, read_result);
      const auto compare_result = std::memcmp(read_cache_line, rw_ops::WRITE_DATA, rw_ops::CACHE_LINE_SIZE);
      // check if all compared bytes were equal.
      ASSERT_EQ(compare_result, 0);
      // Check if data is allocated on the correct numa node.
      ASSERT_EQ(get_numa_node_index_by_address(addr), numa_idx);
    }

    munmap(base_addr, memory_region_size);
  }
}
#endif

}  // namespace mema
