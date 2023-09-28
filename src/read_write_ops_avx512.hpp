#pragma once

#include "read_write_ops_types.hpp"

namespace mema::rw_ops {

#ifdef NT_STORES_AVX_512

#define WRITE_SIMD_NT_512(mem_addr, offset, data) \
  _mm512_stream_si512(reinterpret_cast<__m512i*>((mem_addr) + ((offset)*mema::rw_ops::CACHE_LINE_SIZE)), data)

/**
 * #####################################################
 * NON_TEMPORAL STORE OPERATIONS
 * #####################################################
 */

template <int LOOP_COUNT>
inline void simd_write_nt_64B_accesses(char* address) {
  const auto* data = reinterpret_cast<const __m512i*>(WRITE_DATA);
  // clang-format off
  unroll<LOOP_COUNT>([&](size_t loop_index) {
    WRITE_SIMD_NT_512(address, loop_index, *data);
  });
  // clang-format on
}

template <int LOOP_COUNT>
inline void simd_write_nt_64B_accesses_sfence(char* address) {
  simd_write_nt_64B_accesses<LOOP_COUNT>(address);
  sfence_barrier();
}

#endif  // NT_STORES_AVX_512

}  // namespace mema::rw_ops
