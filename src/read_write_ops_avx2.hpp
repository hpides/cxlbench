#pragma once

#include "read_write_ops_types.hpp"

namespace mema::rw_ops {

#ifdef USE_AVX_2

#define WRITE_SIMD_NT_256(mem_addr, offset, data) \
  _mm256_stream_si256(reinterpret_cast<__m256i*>((mem_addr) + ((offset)*VECTOR_SIZE)), data)

/**
 * #####################################################
 * NON_TEMPORAL STORE OPERATIONS
 * #####################################################
 */

template <int ACCESS_COUNT_64B>
inline void simd_write_nt_64B_accesses(char* address) {
  const auto data = reinterpret_cast<const __m256i*>(WRITE_DATA);
  // clang-format off
  unroll<VECTOR_SIZE_FACTOR * ACCESS_COUNT_64B>([&](size_t loop_index) {
    WRITE_SIMD_NT_256(address, loop_index, *data);
  });
  // clang-format on
}

template <int ACCESS_COUNT_64B>
inline void simd_write_nt_64B_accesses_sfence(char* address) {
  simd_write_nt_64B_accesses<ACCESS_COUNT_64B>(address);
  sfence_barrier();
}

#endif  // USE_AVX_2
}  // namespace mema::rw_ops
