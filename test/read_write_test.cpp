#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <fstream>
#include <read_write_ops.hpp>
#include <utils.hpp>

#include "gtest/gtest.h"
#include "test_utils.hpp"

namespace mema {

constexpr auto MEMORY_REGION_SIZE = size_t{131072};  // 128 KiB

class ReadWriteTest : public BaseTest {
 protected:
  void SetUp() override {
    addr = static_cast<char*>(mmap(nullptr, MEMORY_REGION_SIZE, PROT_READ | PROT_WRITE, utils::MAP_FLAGS, -1, 0));
    ASSERT_NE(addr, MAP_FAILED);
  }

  void TearDown() override { munmap(addr, MEMORY_REGION_SIZE); }

  using MultiWriteFn = void(const std::vector<char*>&);
  void run_multi_write_test(MultiWriteFn write_fn, const size_t access_size) {
    const size_t write_count = MEMORY_REGION_SIZE / access_size;
    const char* last_op = addr + MEMORY_REGION_SIZE;
    std::vector<char*> op_addresses{};
    op_addresses.reserve(write_count);

    for (char* write_addr = addr; write_addr < last_op; write_addr += access_size) {
      op_addresses.emplace_back(write_addr);
    }

    write_fn(op_addresses);
    ASSERT_EQ(msync(addr, MEMORY_REGION_SIZE, MS_SYNC), 0);
  }

  using SingleWriteFn = void(char*);
  void run_single_write_test(SingleWriteFn write_fn, const size_t access_size) {
    write_fn(addr);
    ASSERT_EQ(msync(addr, access_size, MS_SYNC), 0);
  }

  char* addr;
};

TEST_F(ReadWriteTest, SingleSIMDNoneWrite_64) { run_single_write_test(rw_ops::write_none_64, 64); }
TEST_F(ReadWriteTest, SingleSIMDNoneWrite_128) { run_single_write_test(rw_ops::write_none_128, 128); }
TEST_F(ReadWriteTest, SingleSIMDNoneWrite_256) { run_single_write_test(rw_ops::write_none_256, 256); }
TEST_F(ReadWriteTest, SingleSIMDNoneWrite_512) { run_single_write_test(rw_ops::write_none_512, 512); }

TEST_F(ReadWriteTest, MultiSIMDNoneWrite_64) { run_multi_write_test(rw_ops::write_none_64, 64); }
TEST_F(ReadWriteTest, MultiSIMDNoneWrite_128) { run_multi_write_test(rw_ops::write_none_128, 128); }
TEST_F(ReadWriteTest, MultiSIMDNoneWrite_256) { run_multi_write_test(rw_ops::write_none_256, 256); }
TEST_F(ReadWriteTest, MultiSIMDNoneWrite_512) { run_multi_write_test(rw_ops::write_none_512, 512); }

#ifdef HAS_CLWB
TEST_F(ReadWriteTest, SingleSIMDClwbWrite_64) { run_single_write_test(rw_ops::write_clwb_64, 64); }
TEST_F(ReadWriteTest, SingleSIMDClwbWrite_128) { run_single_write_test(rw_ops::write_clwb_128, 128); }
TEST_F(ReadWriteTest, SingleSIMDClwbWrite_256) { run_single_write_test(rw_ops::write_clwb_256, 256); }
TEST_F(ReadWriteTest, SingleSIMDClwbWrite_512) { run_single_write_test(rw_ops::write_clwb_512, 512); }

TEST_F(ReadWriteTest, MultiSIMDClwbWrite_64) { run_multi_write_test(rw_ops::write_clwb_64, 64); }
TEST_F(ReadWriteTest, MultiSIMDClwbWrite_128) { run_multi_write_test(rw_ops::write_clwb_128, 128); }
TEST_F(ReadWriteTest, MultiSIMDClwbWrite_256) { run_multi_write_test(rw_ops::write_clwb_256, 256); }
TEST_F(ReadWriteTest, MultiSIMDClwbWrite_512) { run_multi_write_test(rw_ops::write_clwb_512, 512); }
#endif

#if defined(USE_AVX_2) || defined(USE_AVX_512)
TEST_F(ReadWriteTest, SingleSIMDNonTemporalWrite_64) { run_single_write_test(rw_ops::simd_write_nt_64, 64); }
TEST_F(ReadWriteTest, SingleSIMDNonTemporalWrite_128) { run_single_write_test(rw_ops::simd_write_nt_128, 128); }
TEST_F(ReadWriteTest, SingleSIMDNonTemporalWrite_256) { run_single_write_test(rw_ops::simd_write_nt_256, 256); }
TEST_F(ReadWriteTest, SingleSIMDNonTemporalWrite_512) { run_single_write_test(rw_ops::simd_write_nt_512, 512); }

TEST_F(ReadWriteTest, MultiSIMDNonTemporalWrite_64) { run_multi_write_test(rw_ops::simd_write_nt_64, 64); }
TEST_F(ReadWriteTest, MultiSIMDNonTemporalWrite_128) { run_multi_write_test(rw_ops::simd_write_nt_128, 128); }
TEST_F(ReadWriteTest, MultiSIMDNonTemporalWrite_256) { run_multi_write_test(rw_ops::simd_write_nt_256, 256); }
TEST_F(ReadWriteTest, MultiSIMDNonTemporalWrite_512) { run_multi_write_test(rw_ops::simd_write_nt_512, 512); }
#endif

TEST_F(ReadWriteTest, ScalarWriteTest) {
  constexpr auto thread_count = uint32_t{4};
  auto thread_pool = std::vector<std::thread>{};
  thread_pool.reserve(thread_count);
  auto thread_memory_size = MEMORY_REGION_SIZE / thread_count;

  // Write data.
  for (auto thread_idx = uint8_t{0}; thread_idx < thread_count; thread_idx++) {
    char* from = addr + thread_idx * thread_memory_size;
    const char* to = addr + (thread_idx + 1) * thread_memory_size;
    thread_pool.emplace_back(rw_ops::write_data_scalar, from, to);
  }
  // Wait for all threads.
  for (auto& thread : thread_pool) {
    thread.join();
  }
  // Verify data.
  constexpr auto cache_line_count = MEMORY_REGION_SIZE / rw_ops::CACHE_LINE_SIZE;
  for (auto thread_idx = uint8_t{0}; thread_idx < thread_count; thread_idx++) {
    auto cache_line_address = addr + thread_idx * rw_ops::CACHE_LINE_SIZE;
    auto compare_result = std::memcmp(cache_line_address, rw_ops::WRITE_DATA, rw_ops::CACHE_LINE_SIZE);
    ASSERT_EQ(compare_result, 0);
  }
}

}  // namespace mema
