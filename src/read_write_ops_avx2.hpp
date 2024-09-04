#pragma once

#include "read_write_ops_types.hpp"

namespace mema::rw_ops {

#ifdef USE_AVX_2

#define WRITE_SIMD_NT_256(mem_addr, data) _mm256_stream_si256(reinterpret_cast<__m256i*>(mem_addr), data)

/**
 * #####################################################
 * NON_TEMPORAL STORE OPERATIONS
 * #####################################################
 */

template <int ACCESS_COUNT_64B>
inline void simd_write_nt_64B_accesses(char* address) {
  const auto* data = reinterpret_cast<const __m256i*>(WRITE_DATA);
  constexpr size_t vector_access_count = SIMD_VECTOR_SIZE_FACTOR * ACCESS_COUNT_64B;
#pragma GCC unroll 4096
  for (size_t access_idx = 0; access_idx < vector_access_count; ++access_idx) {
    WRITE_SIMD_NT_256(address + (SIMD_VECTOR_SIZE * access_idx), *data);
  }
}

template <int ACCESS_COUNT_64B>
inline void simd_write_nt_64B_accesses_sfence(char* address) {
  simd_write_nt_64B_accesses<ACCESS_COUNT_64B>(address);
  sfence_barrier();
}

#endif  // USE_AVX_2
}  // namespace mema::rw_ops
