#pragma once

#include <immintrin.h>
#include <xmmintrin.h>

#include <cstdint>
#include <cstring>
#include <vector>

#include "benchmark/benchmark.h"
#include "read_write_ops_avx2.hpp"
#include "read_write_ops_avx512.hpp"

#ifndef HAS_ANY_AVX
#include "read_write_ops_types.hpp"
#endif

namespace mema::rw_ops {

/** flush the cache line using clwb. */
#ifdef HAS_CLWB
inline void flush_clwb(char* addr, const size_t len) {
  const auto* end_addr = addr + len;
  for (auto* current_cl = addr; current_cl < end_addr; current_cl += CACHE_LINE_SIZE) {
    _mm_clwb(current_cl);
  }
}
#endif

#ifdef HAS_ANY_AVX
/**
 * #####################################################
 * STORE OPERATIONS (MULTIPLE ADDRESSES)
 * #####################################################
 */

inline void simd_write_64(const std::vector<char*>& addresses, flush_fn flush, barrier_fn barrier) {
  for (auto* addr : addresses) {
    simd_write_64(addr, flush, barrier);
  }
}

inline void simd_write_128(const std::vector<char*>& addresses, flush_fn flush, barrier_fn barrier) {
  for (auto* addr : addresses) {
    simd_write_128(addr, flush, barrier);
  }
}

inline void simd_write_256(const std::vector<char*>& addresses, flush_fn flush, barrier_fn barrier) {
  for (auto* addr : addresses) {
    simd_write_256(addr, flush, barrier);
  }
}

inline void simd_write_512(const std::vector<char*>& addresses, flush_fn flush, barrier_fn barrier) {
  for (auto* addr : addresses) {
    simd_write_512(addr, flush, barrier);
  }
}

inline void simd_write(const std::vector<char*>& addresses, const size_t access_size, flush_fn flush,
                       barrier_fn barrier) {
  for (auto* addr : addresses) {
    simd_write(addr, access_size, flush, barrier);
  }
}

/**
 * #####################################################
 * STORE + CLWB OPERATIONS
 * #####################################################
 */

#ifdef HAS_CLWB
inline void simd_write_clwb_512(char* addr) { simd_write_512(addr, flush_clwb, sfence_barrier); }

inline void simd_write_clwb_256(char* addr) { simd_write_256(addr, flush_clwb, sfence_barrier); }

inline void simd_write_clwb_128(char* addr) { simd_write_128(addr, flush_clwb, sfence_barrier); }

inline void simd_write_clwb_64(char* addr) { simd_write_64(addr, flush_clwb, sfence_barrier); }

inline void simd_write_clwb(char* addr, const size_t access_size) {
  simd_write(addr, access_size, flush_clwb, sfence_barrier);
}

inline void simd_write_clwb_512(const std::vector<char*>& addresses) {
  simd_write_512(addresses, flush_clwb, sfence_barrier);
}

inline void simd_write_clwb_256(const std::vector<char*>& addresses) {
  simd_write_256(addresses, flush_clwb, sfence_barrier);
}

inline void simd_write_clwb_128(const std::vector<char*>& addresses) {
  simd_write_128(addresses, flush_clwb, sfence_barrier);
}

inline void simd_write_clwb_64(const std::vector<char*>& addresses) {
  simd_write_64(addresses, flush_clwb, sfence_barrier);
}

inline void simd_write_clwb(const std::vector<char*>& addresses, const size_t access_size) {
  simd_write(addresses, access_size, flush_clwb, sfence_barrier);
}

#endif  // clwb

/**
 * #####################################################
 * STORE-ONLY OPERATIONS
 * #####################################################
 */

inline void simd_write_none_512(char* addr) { simd_write_512(addr, no_flush, no_barrier); }

inline void simd_write_none_256(char* addr) { simd_write_256(addr, no_flush, no_barrier); }

inline void simd_write_none_128(char* addr) { simd_write_128(addr, no_flush, no_barrier); }

inline void simd_write_none_64(char* addr) { simd_write_64(addr, no_flush, no_barrier); }

inline void simd_write_none(char* addr, const size_t access_size) {
  simd_write(addr, access_size, no_flush, no_barrier);
}

inline void simd_write_none_512(const std::vector<char*>& addresses) {
  simd_write_512(addresses, no_flush, no_barrier);
}

inline void simd_write_none_256(const std::vector<char*>& addresses) {
  simd_write_256(addresses, no_flush, no_barrier);
}

inline void simd_write_none_128(const std::vector<char*>& addresses) {
  simd_write_128(addresses, no_flush, no_barrier);
}

inline void simd_write_none_64(const std::vector<char*>& addresses) { simd_write_64(addresses, no_flush, no_barrier); }

inline void simd_write_none(const std::vector<char*>& addresses, const size_t access_size) {
  simd_write(addresses, access_size, no_flush, no_barrier);
}

/**
 * #####################################################
 * NON_TEMPORAL STORE OPERATIONS (MULTIPLE ADDRESSES)
 * #####################################################
 */

inline void simd_write_nt_64(const std::vector<char*>& addresses) {
  for (auto* addr : addresses) {
    simd_write_nt_64(addr);
  }
}

inline void simd_write_nt_128(const std::vector<char*>& addresses) {
  for (auto* addr : addresses) {
    simd_write_nt_128(addr);
  }
}

inline void simd_write_nt_256(const std::vector<char*>& addresses) {
  for (auto* addr : addresses) {
    simd_write_nt_256(addr);
  }
}

inline void simd_write_nt_512(const std::vector<char*>& addresses) {
  for (auto* addr : addresses) {
    simd_write_nt_512(addr);
  }
}

inline void simd_write_nt(const std::vector<char*>& addresses, const size_t access_size) {
  for (auto* addr : addresses) {
    simd_write_nt(addr, access_size);
  }
}
#endif  // HAS_ANY_AVX

inline void write_data(char* from, const char* to) {
#ifdef HAS_ANY_AVX
  return simd_write_data_range(from, to);
#endif
}

}  // namespace mema::rw_ops
