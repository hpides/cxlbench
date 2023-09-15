#pragma once

#include "read_write_ops_types.hpp"

namespace mema::rw_ops {

#ifdef HAS_AVX_512

#define READ_SIMD_512(mem_addr, offset) \
  _mm512_load_si512(reinterpret_cast<const void*>((mem_addr) + ((offset)*CACHE_LINE_SIZE)))

#define WRITE_SIMD_NT_512(mem_addr, offset, data) \
  _mm512_stream_si512(reinterpret_cast<__m512i*>((mem_addr) + ((offset)*CACHE_LINE_SIZE)), data)

#define WRITE_SIMD_512(mem_addr, offset, data) \
  _mm512_store_si512(reinterpret_cast<__m512i*>((mem_addr) + ((offset)*CACHE_LINE_SIZE)), data)

inline void simd_write_data_range(char* from, const char* to) {
  const auto* data = reinterpret_cast<const __m512i*>(WRITE_DATA);
  for (char* mem_addr = from; mem_addr < to; mem_addr += CACHE_LINE_SIZE) {
    // Write 512 Bit (64 Byte) and flush it.
    WRITE_SIMD_512(mem_addr, 0, *data);
  }
}

/**
 * #####################################################
 * BASE STORE OPERATIONS
 * #####################################################
 */

inline void simd_write_64(char* addr, flush_fn flush, barrier_fn barrier) {
  // Write 512 Bit (64 Byte)
  const auto* data = reinterpret_cast<const __m512i*>(WRITE_DATA);
  WRITE_SIMD_512(addr, 0, *data);
  flush(addr, 64);
  barrier();
}

inline void simd_write_128(char* addr, flush_fn flush, barrier_fn barrier) {
  // Write 128 Byte
  const auto* data = reinterpret_cast<const __m512i*>(WRITE_DATA);
  WRITE_SIMD_512(addr, 0, *data);
  WRITE_SIMD_512(addr, 1, *data);
  flush(addr, 128);
  barrier();
}

inline void simd_write_256(char* addr, flush_fn flush, barrier_fn barrier) {
  // Write 256 Byte
  const auto* data = reinterpret_cast<const __m512i*>(WRITE_DATA);
  WRITE_SIMD_512(addr, 0, *data);
  WRITE_SIMD_512(addr, 1, *data);
  WRITE_SIMD_512(addr, 2, *data);
  WRITE_SIMD_512(addr, 3, *data);
  flush(addr, 256);
  barrier();
}

inline void simd_write_512(char* addr, flush_fn flush, barrier_fn barrier) {
  // Write 512 Byte
  const auto* data = reinterpret_cast<const __m512i*>(WRITE_DATA);
  WRITE_SIMD_512(addr, 0, *data);
  WRITE_SIMD_512(addr, 1, *data);
  WRITE_SIMD_512(addr, 2, *data);
  WRITE_SIMD_512(addr, 3, *data);
  WRITE_SIMD_512(addr, 4, *data);
  WRITE_SIMD_512(addr, 5, *data);
  WRITE_SIMD_512(addr, 6, *data);
  WRITE_SIMD_512(addr, 7, *data);
  flush(addr, 512);
  barrier();
}

inline void simd_write(char* addr, const size_t access_size, flush_fn flush, barrier_fn barrier) {
  // Write 512 Byte
  const auto* data = reinterpret_cast<const __m512i*>(WRITE_DATA);
  const char* access_end_addr = addr + access_size;
  for (char* mem_addr = addr; mem_addr < access_end_addr; mem_addr += (8 * CACHE_LINE_SIZE)) {
    WRITE_SIMD_512(mem_addr, 0, *data);
    WRITE_SIMD_512(mem_addr, 1, *data);
    WRITE_SIMD_512(mem_addr, 2, *data);
    WRITE_SIMD_512(mem_addr, 3, *data);
    WRITE_SIMD_512(mem_addr, 4, *data);
    WRITE_SIMD_512(mem_addr, 5, *data);
    WRITE_SIMD_512(mem_addr, 6, *data);
    WRITE_SIMD_512(mem_addr, 7, *data);
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
  const auto* data = reinterpret_cast<const __m512i*>(WRITE_DATA);
  WRITE_SIMD_NT_512(addr, 0, *data);
  sfence_barrier();
}

inline void simd_write_nt_128(char* addr) {
  // Write 128 Byte
  const auto* data = reinterpret_cast<const __m512i*>(WRITE_DATA);
  WRITE_SIMD_NT_512(addr, 0, *data);
  WRITE_SIMD_NT_512(addr, 1, *data);
  sfence_barrier();
}

inline void simd_write_nt_256(char* addr) {
  // Write 256 Byte
  const auto* data = reinterpret_cast<const __m512i*>(WRITE_DATA);
  WRITE_SIMD_NT_512(addr, 0, *data);
  WRITE_SIMD_NT_512(addr, 1, *data);
  WRITE_SIMD_NT_512(addr, 2, *data);
  WRITE_SIMD_NT_512(addr, 3, *data);
  sfence_barrier();
}

inline void simd_write_nt_512(char* addr) {
  // Write 512 Byte
  const auto* data = reinterpret_cast<const __m512i*>(WRITE_DATA);
  WRITE_SIMD_NT_512(addr, 0, *data);
  WRITE_SIMD_NT_512(addr, 1, *data);
  WRITE_SIMD_NT_512(addr, 2, *data);
  WRITE_SIMD_NT_512(addr, 3, *data);
  WRITE_SIMD_NT_512(addr, 4, *data);
  WRITE_SIMD_NT_512(addr, 5, *data);
  WRITE_SIMD_NT_512(addr, 6, *data);
  WRITE_SIMD_NT_512(addr, 7, *data);
  sfence_barrier();
}

inline void simd_write_nt(char* addr, const size_t access_size) {
  const auto* data = reinterpret_cast<const __m512i*>(WRITE_DATA);
  const char* access_end_addr = addr + access_size;
  for (char* mem_addr = addr; mem_addr < access_end_addr; mem_addr += (8 * CACHE_LINE_SIZE)) {
    // Write 512 byte.
    WRITE_SIMD_NT_512(mem_addr, 0, *data);
    WRITE_SIMD_NT_512(mem_addr, 1, *data);
    WRITE_SIMD_NT_512(mem_addr, 2, *data);
    WRITE_SIMD_NT_512(mem_addr, 3, *data);
    WRITE_SIMD_NT_512(mem_addr, 4, *data);
    WRITE_SIMD_NT_512(mem_addr, 5, *data);
    WRITE_SIMD_NT_512(mem_addr, 6, *data);
    WRITE_SIMD_NT_512(mem_addr, 7, *data);
  }
  sfence_barrier();
}

/**
 * #####################################################
 * READ OPERATIONS
 * #####################################################
 */

inline __m512i simd_read_64(char* addr) { return READ_SIMD_512(addr, 0); }

inline __m512i simd_read_128(char* addr) {
  __m512i res0{}, res1{};
  res0 = READ_SIMD_512(addr, 0);
  res1 = READ_SIMD_512(addr, 1);
  return res0 + res1;
}

inline __m512i simd_read_256(char* addr) {
  __m512i res0{}, res1{}, res2{}, res3{};
  res0 = READ_SIMD_512(addr, 0);
  res1 = READ_SIMD_512(addr, 1);
  res2 = READ_SIMD_512(addr, 2);
  res3 = READ_SIMD_512(addr, 3);
  return res0 + res1 + res2 + res3;
}

inline __m512i simd_read_512(char* addr) {
  __m512i res0{}, res1{}, res2{}, res3{}, res4{}, res5{}, res6{}, res7{};
  res0 = READ_SIMD_512(addr, 0);
  res1 = READ_SIMD_512(addr, 1);
  res2 = READ_SIMD_512(addr, 2);
  res3 = READ_SIMD_512(addr, 3);
  res4 = READ_SIMD_512(addr, 4);
  res5 = READ_SIMD_512(addr, 5);
  res6 = READ_SIMD_512(addr, 6);
  res7 = READ_SIMD_512(addr, 7);
  return res0 + res1 + res2 + res3 + res4 + res5 + res6 + res7;
}

inline __m512i simd_read(char* addr, const size_t access_size) {
  __m512i res0{}, res1{}, res2{}, res3{}, res4{}, res5{}, res6{}, res7{};
  const char* access_end_addr = addr + access_size;
  for (const char* mem_addr = addr; mem_addr < access_end_addr; mem_addr += (8 * CACHE_LINE_SIZE)) {
    res0 = READ_SIMD_512(addr, 0);
    res1 = READ_SIMD_512(addr, 1);
    res2 = READ_SIMD_512(addr, 2);
    res3 = READ_SIMD_512(addr, 3);
    res4 = READ_SIMD_512(addr, 4);
    res5 = READ_SIMD_512(addr, 5);
    res6 = READ_SIMD_512(addr, 6);
    res7 = READ_SIMD_512(addr, 7);
  }
  return res0 + res1 + res2 + res3 + res4 + res5 + res6 + res7;
}

inline void simd_read_64(const std::vector<char*>& addresses) {
  __m512i res{};
  auto simd_fn = [&]() {
    for (char* addr : addresses) {
      res += simd_read_64(addr);
    }
    return res;
  };
  // Do a single copy of the last read value to the stack from a zmm register. Otherwise, DoNotOptimize copies on each
  // invocation if we have DoNotOptimize in the loop because it cannot be sure how DoNotOptimize modifies the current
  // zmm register.
  __m512i x = simd_fn();
  benchmark::DoNotOptimize(&x);
}

inline void simd_read_128(const std::vector<char*>& addresses) {
  __m512i res0{}, res1{};
  auto simd_fn = [&]() {
    for (char* addr : addresses) {
      // Read 128 Byte
      res0 += READ_SIMD_512(addr, 0);
      res1 += READ_SIMD_512(addr, 1);
    }
    return res0 + res1;
  };
  // Do a single copy of the last read value to the stack from a zmm register. Otherwise, DoNotOptimize copies on each
  // invocation if we have DoNotOptimize in the loop because it cannot be sure how DoNotOptimize modifies the current
  // zmm register.
  __m512i x = simd_fn();
  benchmark::DoNotOptimize(&x);
}

inline void simd_read_256(const std::vector<char*>& addresses) {
  __m512i res0{}, res1{}, res2{}, res3{};
  auto simd_fn = [&]() {
    for (char* addr : addresses) {
      // Read 256 Byte
      res0 += READ_SIMD_512(addr, 0);
      res1 += READ_SIMD_512(addr, 1);
      res2 += READ_SIMD_512(addr, 2);
      res3 += READ_SIMD_512(addr, 3);
    }
    return res0 + res1 + res2 + res3;
  };
  // Do a single copy of the last read value to the stack from a zmm register. Otherwise, DoNotOptimize copies on each
  // invocation if we have DoNotOptimize in the loop because it cannot be sure how DoNotOptimize modifies the current
  // zmm register.
  __m512i x = simd_fn();
  benchmark::DoNotOptimize(&x);
}

inline void simd_read_512(const std::vector<char*>& addresses) {
  __m512i res0{}, res1{}, res2{}, res3{}, res4{}, res5{}, res6{}, res7{};
  auto simd_fn = [&]() {
    for (char* addr : addresses) {
      // Read 512 Byte
      res0 += READ_SIMD_512(addr, 0);
      res1 += READ_SIMD_512(addr, 1);
      res2 += READ_SIMD_512(addr, 2);
      res3 += READ_SIMD_512(addr, 3);
      res4 += READ_SIMD_512(addr, 4);
      res5 += READ_SIMD_512(addr, 5);
      res6 += READ_SIMD_512(addr, 6);
      res7 += READ_SIMD_512(addr, 7);
    }
    return res0 + res1 + res2 + res3 + res4 + res5 + res6 + res7;
  };
  // Do a single copy of the last read value to the stack from a zmm register. Otherwise, DoNotOptimize copies on each
  // invocation if we have DoNotOptimize in the loop because it cannot be sure how DoNotOptimize modifies the current
  // zmm register.
  __m512i x = simd_fn();
  benchmark::DoNotOptimize(&x);
}

inline void simd_read(const std::vector<char*>& addresses, const size_t access_size) {
  __m512i res0, res1, res2, res3, res4, res5, res6, res7;
  auto simd_fn = [&]() {
    for (char* addr : addresses) {
      const char* access_end_addr = addr + access_size;
      for (const char* mem_addr = addr; mem_addr < access_end_addr; mem_addr += (8 * CACHE_LINE_SIZE)) {
        // Read in 512 Byte chunks
        res0 += READ_SIMD_512(mem_addr, 0);
        res1 += READ_SIMD_512(mem_addr, 1);
        res2 += READ_SIMD_512(mem_addr, 2);
        res3 += READ_SIMD_512(mem_addr, 3);
        res4 += READ_SIMD_512(mem_addr, 4);
        res5 += READ_SIMD_512(mem_addr, 5);
        res6 += READ_SIMD_512(mem_addr, 6);
        res7 += READ_SIMD_512(mem_addr, 7);
      }
    }
    return res0 + res1 + res2 + res3 + res4 + res5 + res6 + res7;
  };
  // Do a single copy of the last read value to the stack from a zmm register. Otherwise, DoNotOptimize copies on each
  // invocation if we have DoNotOptimize in the loop because it cannot be sure how DoNotOptimize modifies the current
  // zmm register.
  __m512i x = simd_fn();
  benchmark::DoNotOptimize(&x);
}

#endif  // HAS_AVX_512

}  // namespace mema::rw_ops
