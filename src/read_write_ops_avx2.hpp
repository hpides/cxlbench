#pragma once

#include "read_write_ops_types.hpp"

namespace mema::rw_ops {

#ifdef HAS_AVX_2

#define READ_SIMD_256(mem_addr, offset) _mm256_load_si256(reinterpret_cast<const __m256i*>((mem_addr) + ((offset)*32)))

#define WRITE_SIMD_NT_256(mem_addr, offset, data) \
  _mm256_stream_si256(reinterpret_cast<__m256i*>((mem_addr) + ((offset)*32)), data)

#define WRITE_SIMD_256(mem_addr, offset, data) \
  _mm256_store_si256(reinterpret_cast<__m256i*>((mem_addr) + ((offset)*32)), data)

inline void simd_write_data_range(char* from, const char* to) {
  const auto data_chunk_1 = reinterpret_cast<const __m256i*>(WRITE_DATA);
  const auto data_chunk_2 = reinterpret_cast<const __m256i*>(WRITE_DATA + 32);
  for (char* mem_addr = from; mem_addr < to; mem_addr += CACHE_LINE_SIZE) {
    // Write 2 x 256 Bit (2 x 32 Byte).
    WRITE_SIMD_256(mem_addr, 0, *data_chunk_1);
    WRITE_SIMD_256(mem_addr, 1, *data_chunk_2);
  }
}

/**
 * #####################################################
 * BASE STORE OPERATIONS
 * #####################################################
 */

inline void simd_write_64(char* addr, flush_fn flush, barrier_fn barrier) {
  // Write 2 x 256 Bit (2 x 32 Byte).
  const auto data_chunk_1 = reinterpret_cast<const __m256i*>(WRITE_DATA);
  const auto data_chunk_2 = reinterpret_cast<const __m256i*>(WRITE_DATA + 32);
  WRITE_SIMD_256(addr, 0, *data_chunk_1);
  WRITE_SIMD_256(addr, 1, *data_chunk_2);
  flush(addr, 64);
  barrier();
}

inline void simd_write_128(char* addr, flush_fn flush, barrier_fn barrier) {
  // Write 128 Byte
  const auto data_chunk_1 = reinterpret_cast<const __m256i*>(WRITE_DATA);
  const auto data_chunk_2 = reinterpret_cast<const __m256i*>(WRITE_DATA + 32);
  WRITE_SIMD_256(addr, 0, *data_chunk_1);
  WRITE_SIMD_256(addr, 1, *data_chunk_2);
  WRITE_SIMD_256(addr, 2, *data_chunk_1);
  WRITE_SIMD_256(addr, 3, *data_chunk_2);
  flush(addr, 128);
  barrier();
}

inline void simd_write_256(char* addr, flush_fn flush, barrier_fn barrier) {
  // Write 256 Byte
  const auto data_chunk_1 = reinterpret_cast<const __m256i*>(WRITE_DATA);
  const auto data_chunk_2 = reinterpret_cast<const __m256i*>(WRITE_DATA + 32);
  WRITE_SIMD_256(addr, 0, *data_chunk_1);
  WRITE_SIMD_256(addr, 1, *data_chunk_2);
  WRITE_SIMD_256(addr, 2, *data_chunk_1);
  WRITE_SIMD_256(addr, 3, *data_chunk_2);
  WRITE_SIMD_256(addr, 4, *data_chunk_1);
  WRITE_SIMD_256(addr, 5, *data_chunk_2);
  WRITE_SIMD_256(addr, 6, *data_chunk_1);
  WRITE_SIMD_256(addr, 7, *data_chunk_2);
  flush(addr, 256);
  barrier();
}

inline void simd_write_512(char* addr, flush_fn flush, barrier_fn barrier) {
  // Write 512 Byte
  const auto data_chunk_1 = reinterpret_cast<const __m256i*>(WRITE_DATA);
  const auto data_chunk_2 = reinterpret_cast<const __m256i*>(WRITE_DATA + 32);
  WRITE_SIMD_256(addr, 0, *data_chunk_1);
  WRITE_SIMD_256(addr, 1, *data_chunk_2);
  WRITE_SIMD_256(addr, 2, *data_chunk_1);
  WRITE_SIMD_256(addr, 3, *data_chunk_2);
  WRITE_SIMD_256(addr, 4, *data_chunk_1);
  WRITE_SIMD_256(addr, 5, *data_chunk_2);
  WRITE_SIMD_256(addr, 6, *data_chunk_1);
  WRITE_SIMD_256(addr, 7, *data_chunk_2);
  WRITE_SIMD_256(addr, 8, *data_chunk_1);
  WRITE_SIMD_256(addr, 9, *data_chunk_2);
  WRITE_SIMD_256(addr, 10, *data_chunk_1);
  WRITE_SIMD_256(addr, 11, *data_chunk_2);
  WRITE_SIMD_256(addr, 12, *data_chunk_1);
  WRITE_SIMD_256(addr, 13, *data_chunk_2);
  WRITE_SIMD_256(addr, 14, *data_chunk_1);
  WRITE_SIMD_256(addr, 15, *data_chunk_2);
  flush(addr, 512);
  barrier();
}

inline void simd_write(char* addr, const size_t access_size, flush_fn flush, barrier_fn barrier) {
  const auto data_chunk_1 = reinterpret_cast<const __m256i*>(WRITE_DATA);
  const auto data_chunk_2 = reinterpret_cast<const __m256i*>(WRITE_DATA + 32);
  const char* access_end_addr = addr + access_size;
  for (char* mem_addr = addr; mem_addr < access_end_addr; mem_addr += (8 * CACHE_LINE_SIZE)) {
    // Write 512 Byte
    WRITE_SIMD_256(mem_addr, 0, *data_chunk_1);
    WRITE_SIMD_256(mem_addr, 1, *data_chunk_2);
    WRITE_SIMD_256(mem_addr, 2, *data_chunk_1);
    WRITE_SIMD_256(mem_addr, 3, *data_chunk_2);
    WRITE_SIMD_256(mem_addr, 4, *data_chunk_1);
    WRITE_SIMD_256(mem_addr, 5, *data_chunk_2);
    WRITE_SIMD_256(mem_addr, 6, *data_chunk_1);
    WRITE_SIMD_256(mem_addr, 7, *data_chunk_2);
    WRITE_SIMD_256(mem_addr, 8, *data_chunk_1);
    WRITE_SIMD_256(mem_addr, 9, *data_chunk_2);
    WRITE_SIMD_256(mem_addr, 10, *data_chunk_1);
    WRITE_SIMD_256(mem_addr, 11, *data_chunk_2);
    WRITE_SIMD_256(mem_addr, 12, *data_chunk_1);
    WRITE_SIMD_256(mem_addr, 13, *data_chunk_2);
    WRITE_SIMD_256(mem_addr, 14, *data_chunk_1);
    WRITE_SIMD_256(mem_addr, 15, *data_chunk_2);
  }
  flush(addr, access_size);
  barrier();
}

/**
 * #####################################################
 * NON_TEMPORAL STORE OPERATIONS
 * #####################################################
 */

inline void simd_write_nt_64(char* addr) {
  // Write 512 Bit (64 Byte)
  const auto data_chunk_1 = reinterpret_cast<const __m256i*>(WRITE_DATA);
  const auto data_chunk_2 = reinterpret_cast<const __m256i*>(WRITE_DATA + 32);
  WRITE_SIMD_NT_256(addr, 0, *data_chunk_1);
  WRITE_SIMD_NT_256(addr, 1, *data_chunk_2);
  sfence_barrier();
}

inline void simd_write_nt_128(char* addr) {
  // Write 128 Byte
  const auto data_chunk_1 = reinterpret_cast<const __m256i*>(WRITE_DATA);
  const auto data_chunk_2 = reinterpret_cast<const __m256i*>(WRITE_DATA + 32);
  WRITE_SIMD_NT_256(addr, 0, *data_chunk_1);
  WRITE_SIMD_NT_256(addr, 1, *data_chunk_2);
  WRITE_SIMD_NT_256(addr, 2, *data_chunk_1);
  WRITE_SIMD_NT_256(addr, 3, *data_chunk_2);
  sfence_barrier();
}

inline void simd_write_nt_256(char* addr) {
  // Write 256 Byte
  const auto data_chunk_1 = reinterpret_cast<const __m256i*>(WRITE_DATA);
  const auto data_chunk_2 = reinterpret_cast<const __m256i*>(WRITE_DATA + 32);
  WRITE_SIMD_NT_256(addr, 0, *data_chunk_1);
  WRITE_SIMD_NT_256(addr, 1, *data_chunk_2);
  WRITE_SIMD_NT_256(addr, 2, *data_chunk_1);
  WRITE_SIMD_NT_256(addr, 3, *data_chunk_2);
  WRITE_SIMD_NT_256(addr, 4, *data_chunk_1);
  WRITE_SIMD_NT_256(addr, 5, *data_chunk_2);
  WRITE_SIMD_NT_256(addr, 6, *data_chunk_1);
  WRITE_SIMD_NT_256(addr, 7, *data_chunk_2);
  sfence_barrier();
}

inline void simd_write_nt_512(char* addr) {
  // Write 512 Byte
  const auto data_chunk_1 = reinterpret_cast<const __m256i*>(WRITE_DATA);
  const auto data_chunk_2 = reinterpret_cast<const __m256i*>(WRITE_DATA + 32);
  WRITE_SIMD_NT_256(addr, 0, *data_chunk_1);
  WRITE_SIMD_NT_256(addr, 1, *data_chunk_2);
  WRITE_SIMD_NT_256(addr, 2, *data_chunk_1);
  WRITE_SIMD_NT_256(addr, 3, *data_chunk_2);
  WRITE_SIMD_NT_256(addr, 4, *data_chunk_1);
  WRITE_SIMD_NT_256(addr, 5, *data_chunk_2);
  WRITE_SIMD_NT_256(addr, 6, *data_chunk_1);
  WRITE_SIMD_NT_256(addr, 7, *data_chunk_2);
  WRITE_SIMD_NT_256(addr, 8, *data_chunk_1);
  WRITE_SIMD_NT_256(addr, 9, *data_chunk_2);
  WRITE_SIMD_NT_256(addr, 10, *data_chunk_1);
  WRITE_SIMD_NT_256(addr, 11, *data_chunk_2);
  WRITE_SIMD_NT_256(addr, 12, *data_chunk_1);
  WRITE_SIMD_NT_256(addr, 13, *data_chunk_2);
  WRITE_SIMD_NT_256(addr, 14, *data_chunk_1);
  WRITE_SIMD_NT_256(addr, 15, *data_chunk_2);
  sfence_barrier();
}

inline void simd_write_nt(char* addr, const size_t access_size) {
  const auto data_chunk_1 = reinterpret_cast<const __m256i*>(WRITE_DATA);
  const auto data_chunk_2 = reinterpret_cast<const __m256i*>(WRITE_DATA + 32);
  const char* access_end_addr = addr + access_size;
  for (char* mem_addr = addr; mem_addr < access_end_addr; mem_addr += (8 * CACHE_LINE_SIZE)) {
    // Write 512 byte.
    WRITE_SIMD_NT_256(mem_addr, 0, *data_chunk_1);
    WRITE_SIMD_NT_256(mem_addr, 1, *data_chunk_2);
    WRITE_SIMD_NT_256(mem_addr, 2, *data_chunk_1);
    WRITE_SIMD_NT_256(mem_addr, 3, *data_chunk_2);
    WRITE_SIMD_NT_256(mem_addr, 4, *data_chunk_1);
    WRITE_SIMD_NT_256(mem_addr, 5, *data_chunk_2);
    WRITE_SIMD_NT_256(mem_addr, 6, *data_chunk_1);
    WRITE_SIMD_NT_256(mem_addr, 7, *data_chunk_2);
    WRITE_SIMD_NT_256(mem_addr, 8, *data_chunk_1);
    WRITE_SIMD_NT_256(mem_addr, 9, *data_chunk_2);
    WRITE_SIMD_NT_256(mem_addr, 10, *data_chunk_1);
    WRITE_SIMD_NT_256(mem_addr, 11, *data_chunk_2);
    WRITE_SIMD_NT_256(mem_addr, 12, *data_chunk_1);
    WRITE_SIMD_NT_256(mem_addr, 13, *data_chunk_2);
    WRITE_SIMD_NT_256(mem_addr, 14, *data_chunk_1);
    WRITE_SIMD_NT_256(mem_addr, 15, *data_chunk_2);
  }
  sfence_barrier();
}

/**
 * #####################################################
 * READ OPERATIONS
 * #####################################################
 */

inline __m256i simd_read_64(char* addr) {
  __m256i res[2];
  res[0] = READ_SIMD_256(addr, 0);
  res[1] = READ_SIMD_256(addr, 1);
  return res[0] + res[1];
}

inline __m256i simd_read_128(char* addr) {
  __m256i res[4];
  res[0] = READ_SIMD_256(addr, 0);
  res[1] = READ_SIMD_256(addr, 1);
  res[2] = READ_SIMD_256(addr, 2);
  res[3] = READ_SIMD_256(addr, 3);
  return res[0] + res[1] + res[2] + res[3];
}

inline __m256i simd_read_256(char* addr) {
  __m256i res[8];
  res[0] = READ_SIMD_256(addr, 0);
  res[1] = READ_SIMD_256(addr, 1);
  res[2] = READ_SIMD_256(addr, 2);
  res[3] = READ_SIMD_256(addr, 3);
  res[4] = READ_SIMD_256(addr, 4);
  res[5] = READ_SIMD_256(addr, 5);
  res[6] = READ_SIMD_256(addr, 6);
  res[7] = READ_SIMD_256(addr, 7);
  return res[0] + res[1] + res[2] + res[3] + res[4] + res[5] + res[6] + res[7];
}

inline __m256i simd_read_512(char* addr) {
  __m256i res[16];
  res[0] = READ_SIMD_256(addr, 0);
  res[1] = READ_SIMD_256(addr, 1);
  res[2] = READ_SIMD_256(addr, 2);
  res[3] = READ_SIMD_256(addr, 3);
  res[4] = READ_SIMD_256(addr, 4);
  res[5] = READ_SIMD_256(addr, 5);
  res[6] = READ_SIMD_256(addr, 6);
  res[7] = READ_SIMD_256(addr, 7);
  res[8] = READ_SIMD_256(addr, 8);
  res[9] = READ_SIMD_256(addr, 9);
  res[10] = READ_SIMD_256(addr, 10);
  res[11] = READ_SIMD_256(addr, 11);
  res[12] = READ_SIMD_256(addr, 12);
  res[13] = READ_SIMD_256(addr, 13);
  res[14] = READ_SIMD_256(addr, 14);
  res[15] = READ_SIMD_256(addr, 15);
  return res[0] + res[1] + res[2] + res[3] + res[4] + res[5] + res[6] + res[7] + res[8] + res[9] + res[10] + res[11] +
         res[12] + res[13] + res[14] + res[15];
}

inline __m256i simd_read(char* addr, const size_t access_size) {
  __m256i res[16];
  const char* access_end_addr = addr + access_size;
  for (const char* mem_addr = addr; mem_addr < access_end_addr; mem_addr += (8 * CACHE_LINE_SIZE)) {
    res[0] = READ_SIMD_256(addr, 0);
    res[1] = READ_SIMD_256(addr, 1);
    res[2] = READ_SIMD_256(addr, 2);
    res[3] = READ_SIMD_256(addr, 3);
    res[4] = READ_SIMD_256(addr, 4);
    res[5] = READ_SIMD_256(addr, 5);
    res[6] = READ_SIMD_256(addr, 6);
    res[7] = READ_SIMD_256(addr, 7);
    res[8] = READ_SIMD_256(addr, 8);
    res[9] = READ_SIMD_256(addr, 9);
    res[10] = READ_SIMD_256(addr, 10);
    res[11] = READ_SIMD_256(addr, 11);
    res[12] = READ_SIMD_256(addr, 12);
    res[13] = READ_SIMD_256(addr, 13);
    res[14] = READ_SIMD_256(addr, 14);
    res[15] = READ_SIMD_256(addr, 15);
  }
  return res[0] + res[1] + res[2] + res[3] + res[4] + res[5] + res[6] + res[7] + res[8] + res[9] + res[10] + res[11] +
         res[12] + res[13] + res[14] + res[15];
}

inline void simd_read_64(const std::vector<char*>& addresses) {
  __m256i res{};
  auto simd_fn = [&]() {
    for (char* addr : addresses) {
      res += simd_read_64(addr);
    }
    return res;
  };
  // Do a single copy of the last read value to the stack from a zmm register. Otherwise, DoNotOptimize copies on each
  // invocation if we have DoNotOptimize in the loop because it cannot be sure how DoNotOptimize modifies the current
  // zmm register.
  __m256i x = simd_fn();
  benchmark::DoNotOptimize(&x);
}

inline void simd_read_128(const std::vector<char*>& addresses) {
  __m256i res{};
  auto simd_fn = [&]() {
    for (char* addr : addresses) {
      res += simd_read_128(addr);
    }
    return res;
  };
  // Do a single copy of the last read value to the stack from a zmm register. Otherwise, DoNotOptimize copies on each
  // invocation if we have DoNotOptimize in the loop because it cannot be sure how DoNotOptimize modifies the current
  // zmm register.
  __m256i x = simd_fn();
  benchmark::DoNotOptimize(&x);
}

inline void simd_read_256(const std::vector<char*>& addresses) {
  __m256i res{};
  auto simd_fn = [&]() {
    for (char* addr : addresses) {
      res += simd_read_256(addr);
    }
    return res;
  };
  // Do a single copy of the last read value to the stack from a zmm register. Otherwise, DoNotOptimize copies on each
  // invocation if we have DoNotOptimize in the loop because it cannot be sure how DoNotOptimize modifies the current
  // zmm register.
  __m256i x = simd_fn();
  benchmark::DoNotOptimize(&x);
}

inline void simd_read_512(const std::vector<char*>& addresses) {
  __m256i res{};
  auto simd_fn = [&]() {
    for (char* addr : addresses) {
      res += simd_read_512(addr);
    }
    return res;
  };
  // Do a single copy of the last read value to the stack from a zmm register. Otherwise, DoNotOptimize copies on each
  // invocation if we have DoNotOptimize in the loop because it cannot be sure how DoNotOptimize modifies the current
  // zmm register.
  __m256i x = simd_fn();
  benchmark::DoNotOptimize(&x);
}

inline void simd_read(const std::vector<char*>& addresses, const size_t access_size) {
  __m256i res{};
  auto simd_fn = [&]() {
    for (char* addr : addresses) {
      const char* access_end_addr = addr + access_size;
      for (const char* mem_addr = addr; mem_addr < access_end_addr; mem_addr += (8 * CACHE_LINE_SIZE)) {
        // Read in 512 Byte chunks
        res += simd_read_512(addr);
      }
    }
    return res;
  };
  // Do a single copy of the last read value to the stack from a zmm register. Otherwise, DoNotOptimize copies on each
  // invocation if we have DoNotOptimize in the loop because it cannot be sure how DoNotOptimize modifies the current
  // zmm register.
  __m256i x = simd_fn();
  benchmark::DoNotOptimize(&x);
}

#endif  // HAS_AVX_2
}  // namespace mema::rw_ops
