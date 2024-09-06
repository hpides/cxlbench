#pragma once

#include "read_write_ops_types.hpp"

namespace cxlbench::rw_ops {

#ifdef USE_AVX_512

#define WRITE_STREAM_512(mem_addr, data) _mm512_stream_si512(reinterpret_cast<__m512i*>(mem_addr), data)

#define READ_STREAM_512(mem_addr) _mm512_stream_load_si512(reinterpret_cast<__m512i*>(mem_addr))

/**
 * #####################################################
 * NON_TEMPORAL LOAD OPERATIONS
 * #####################################################
 */

inline char read_stream_64B_access(char* address) {
  static volatile auto result = __m512i{};
  result = READ_STREAM_512(address);
  const auto* char_res = reinterpret_cast<volatile char*>(&result);
  return char_res[0];
}

template <int ACCESS_COUNT_64B>
inline char read_stream_64B_accesses(char* address) {
  static volatile auto result = __m512i{};
#pragma GCC unroll 1024
  for (size_t access_idx = 0; access_idx < ACCESS_COUNT_64B; ++access_idx) {
    auto* next_addr = address + (access_idx * 64);
    result = READ_STREAM_512(next_addr);
  }
  const auto* char_res = reinterpret_cast<volatile char*>(&result);
  return char_res[0];
}

inline u64 read_64_stream_get_u64(char* addr) {
  volatile auto result = __m512i{};
  result = READ_STREAM_512(addr);
  const auto* char_res = reinterpret_cast<volatile char*>(&result);
  const auto* u64_res = reinterpret_cast<const volatile u64*>(char_res);
  return u64_res[0];
}

/**
 * #####################################################
 * NON_TEMPORAL STORE OPERATIONS
 * #####################################################
 */

template <int ACCESS_COUNT_64B>
inline void write_stream_64B_accesses(char* address) {
  const auto* data = reinterpret_cast<const __m512i*>(WRITE_DATA);
#pragma GCC unroll 4096
  for (size_t access_idx = 0; access_idx < ACCESS_COUNT_64B; ++access_idx) {
    auto next_addr = address + (access_idx * 64);
    WRITE_STREAM_512(next_addr, *data);
  }
}

inline void write_stream_64B(char* address) {
  const auto* data = reinterpret_cast<const __m512i*>(WRITE_DATA);
  WRITE_STREAM_512(address, *data);
}

#endif  // USE_AVX_512

}  // namespace cxlbench::rw_ops
