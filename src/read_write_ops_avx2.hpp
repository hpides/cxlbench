#pragma once

#include "read_write_ops_types.hpp"

namespace mema::rw_ops {

#ifdef USE_AVX_2

#define WRITE_SIMD_NT_256(mem_addr, offset, data) \
  _mm256_stream_si256(reinterpret_cast<__m256i*>((mem_addr) + ((offset)*32)), data)

/**
 * #####################################################
 * NON_TEMPORAL STORE OPERATIONS
 * #####################################################
 */

template <int LOOP_COUNT>
inline void simd_write_nt_64B_accesses(char* address) {
  const auto data_chunk_1 = reinterpret_cast<const __m256i*>(WRITE_DATA);
  const auto data_chunk_2 = reinterpret_cast<const __m256i*>(WRITE_DATA + 32);
  // clang-format off
  unroll<LOOP_COUNT>([&](size_t loop_index) {
    WRITE_SIMD_NT_256(address, loop_index, *data_chunk_1);
    WRITE_SIMD_NT_256(address, loop_index + 1, *data_chunk_2);
  });
  // clang-format on
}

template <int LOOP_COUNT>
inline void simd_write_nt_64B_accesses_sfence(char* address) {
  simd_write_nt_64B_accesses<LOOP_COUNT>(address);
  sfence_barrier();
}

#endif  // USE_AVX_2
}  // namespace mema::rw_ops
