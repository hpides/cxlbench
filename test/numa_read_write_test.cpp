#include <cstring>
#include <cstddef>
#include <iostream>
#include <string>
#include <unordered_set>

#include <immintrin.h>
#include <numa.h>
#include <string.h>
#include <sys/mman.h>

#include "read_write_ops.hpp"
#include "numa.hpp"
#include "utils.hpp"

#include "gtest/gtest.h"

namespace {
  constexpr uint32_t MIB_IN_BYTES = 1024 * 1024;
}

namespace mema {
  
class NumaReadWriteTest : public ::testing::Test {
 protected:
  void SetUp() override {
    init_numa({});
  }
};

TEST_F(NumaReadWriteTest, SimpleWriteRead) {
  const auto memory_region_size = 100 * MIB_IN_BYTES;
  const auto num_numa_nodes = numa_num_configured_nodes();
  for (auto numa_idx = NumaNodeID{0}; numa_idx < num_numa_nodes; ++numa_idx) {
    SCOPED_TRACE("NUMA node index: " + std::to_string(numa_idx));
    char* base_addr = static_cast<char*>(mmap(nullptr, memory_region_size, PROT_READ | PROT_WRITE, mema::utils::DRAM_MAP_FLAGS, -1, 0));
    ASSERT_NE(base_addr, MAP_FAILED);
    ASSERT_NE(base_addr, nullptr);
    set_memory_on_numa_nodes(base_addr, memory_region_size, {numa_idx});
    const auto cache_line_count = memory_region_size / mema::rw_ops::CACHE_LINE_SIZE;
    auto addresses = std::unordered_set<char*>{};
    for (auto cache_line_idx = size_t{0}; cache_line_idx < cache_line_count; ++cache_line_idx) {        
      const auto addr = base_addr + (cache_line_idx * mema::rw_ops::CACHE_LINE_SIZE);
      std::memcpy(addr, mema::rw_ops::WRITE_DATA, mema::rw_ops::CACHE_LINE_SIZE);
      const auto compare_result = std::memcmp(addr, mema::rw_ops::WRITE_DATA, mema::rw_ops::CACHE_LINE_SIZE);
      // check if all compared bytes were equal.
      ASSERT_EQ(compare_result, 0);
      ASSERT_FALSE((addresses.contains(addr)));
      addresses.insert(addr);
      ASSERT_TRUE((addresses.contains(addr)));
    }
    munmap(base_addr, memory_region_size);
  }
}

#ifdef HAS_AVX
TEST_F(NumaReadWriteTest, AVX512Read) {
  const auto memory_region_size = 100 * MIB_IN_BYTES;
  const auto num_numa_nodes = numa_num_configured_nodes();
  for (auto numa_idx = NumaNodeID{0}; numa_idx < num_numa_nodes; ++numa_idx) {
    SCOPED_TRACE("NUMA node index: " + std::to_string(numa_idx));
    char* base_addr = static_cast<char*>(mmap(nullptr, memory_region_size, PROT_READ | PROT_WRITE, mema::utils::DRAM_MAP_FLAGS, -1, 0));
    ASSERT_NE(base_addr, MAP_FAILED);
    ASSERT_NE(base_addr, nullptr);
    set_memory_on_numa_nodes(base_addr, memory_region_size, {numa_idx});
    const auto cache_line_count = memory_region_size / mema::rw_ops::CACHE_LINE_SIZE;
    for (auto cache_line_idx = size_t{0}; cache_line_idx < cache_line_count; ++cache_line_idx) {        
      const auto addr = base_addr + (cache_line_idx * mema::rw_ops::CACHE_LINE_SIZE);
      // write data to memory region
      std::memcpy(addr, mema::rw_ops::WRITE_DATA, mema::rw_ops::CACHE_LINE_SIZE);
      // read data from memory region via AVX512 intrinsics into SIMD registers
      const __m512i read_result = mema::rw_ops::simd_read_64(addr);
      // store data from SIMD registers into local char array
      char read_cache_line[mema::rw_ops::CACHE_LINE_SIZE] __attribute__((aligned(64))) = {};
      _mm512_store_si512(read_cache_line, read_result);
      const auto compare_result = std::memcmp(read_cache_line, mema::rw_ops::WRITE_DATA, mema::rw_ops::CACHE_LINE_SIZE);
      // check if all compared bytes were equal.
      ASSERT_EQ(compare_result, 0);
    }
    munmap(base_addr, memory_region_size);
  }
}

TEST_F(NumaReadWriteTest, AVX512WriteRead) {
  const auto memory_region_size = 100 * MIB_IN_BYTES;
  const auto num_numa_nodes = numa_num_configured_nodes();
  for (auto numa_idx = NumaNodeID{0}; numa_idx < num_numa_nodes; ++numa_idx) {
    SCOPED_TRACE("NUMA node index: " + std::to_string(numa_idx));
    char* base_addr = static_cast<char*>(mmap(nullptr, memory_region_size, PROT_READ | PROT_WRITE, mema::utils::DRAM_MAP_FLAGS, -1, 0));
    ASSERT_NE(base_addr, MAP_FAILED);
    ASSERT_NE(base_addr, nullptr);
    set_memory_on_numa_nodes(base_addr, memory_region_size, {numa_idx});
    const auto cache_line_count = memory_region_size / mema::rw_ops::CACHE_LINE_SIZE;
    for (auto cache_line_idx = size_t{0}; cache_line_idx < cache_line_count; ++cache_line_idx) {        
      const auto addr = base_addr + (cache_line_idx * mema::rw_ops::CACHE_LINE_SIZE);
      // write data to memory region via AVX512 intrinsics
      mema:rw_ops::simd_write_none_64(addr);
      // read data from memory region via AVX512 intrinsics into SIMD registers
      const __m512i read_result = mema::rw_ops::simd_read_64(addr);
      // store data from SIMD registers into local char array
      char read_cache_line[mema::rw_ops::CACHE_LINE_SIZE] __attribute__((aligned(64))) = {};
      _mm512_store_si512(read_cache_line, read_result);
      const auto compare_result = std::memcmp(read_cache_line, mema::rw_ops::WRITE_DATA, mema::rw_ops::CACHE_LINE_SIZE);
      // check if all compared bytes were equal.
      ASSERT_EQ(compare_result, 0);
    }
    munmap(base_addr, memory_region_size);
  }
}
#endif

} // namespace mema
