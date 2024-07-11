#pragma once

#include <cstdint>
#include <cstring>
#include <vector>

#include "benchmark/benchmark.h"
#include "read_write_ops_avx2.hpp"
#include "read_write_ops_avx512.hpp"
#if !(defined(USE_AVX_2) || defined(USE_AVX_512))
#include "read_write_ops_types.hpp"
#endif

/** Neon intrinsics for ARM */
#if defined(__ARM_NEON)
#include <arm_neon.h>
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

/**
 * #####################################################
 * READ OPERATIONS
 * #####################################################
 */

template <int ACCESS_COUNT_64B>
inline CharVecBase read_64B_accesses(char* address) {
  volatile CharVecSIMD* volatile_addr = reinterpret_cast<CharVecSIMD*>(address);
  auto result = CharVecBase{0};
  auto result_vec_simd = reinterpret_cast<CharVecSIMD*>(&result);
  // The maximum access size is 64 KiB. With a 64 B base access size, we need 1024 accesses.
#pragma GCC unroll 1024
  for (auto base_access_idx = uint64_t{0}; base_access_idx < ACCESS_COUNT_64B; ++base_access_idx) {
// Perform base access with SIMD_VECTOR_SIZE_FACTOR * vector accesses.
#pragma GCC unroll SIMD_VECTOR_SIZE_FACTOR
    for (auto sub_access_idx = uint64_t{0}; sub_access_idx < SIMD_VECTOR_SIZE_FACTOR; ++sub_access_idx) {
      const auto index = base_access_idx * SIMD_VECTOR_SIZE_FACTOR + sub_access_idx;
      result_vec_simd[index] = volatile_addr[index];
    }
  }
  return result;
}

inline CharVecBase read_64(char* addr) { return read_64B_accesses<1>(addr); }

inline CharVecBase read_128(char* addr) { return read_64B_accesses<2>(addr); }

inline CharVecBase read_256(char* addr) { return read_64B_accesses<4>(addr); }

inline CharVecBase read_512(char* addr) { return read_64B_accesses<8>(addr); }

inline CharVecBase read_1k(char* addr) { return read_64B_accesses<16>(addr); }

inline CharVecBase read_2k(char* addr) { return read_64B_accesses<32>(addr); }

inline CharVecBase read_4k(char* addr) { return read_64B_accesses<64>(addr); }

inline CharVecBase read_8k(char* addr) { return read_64B_accesses<128>(addr); }

inline CharVecBase read_16k(char* addr) { return read_64B_accesses<256>(addr); }

inline CharVecBase read_32k(char* addr) { return read_64B_accesses<512>(addr); }

inline CharVecBase read_64k(char* addr) { return read_64B_accesses<1024>(addr); }

inline CharVecBase read(char* addr, const size_t access_size) {
  auto result = CharVecBase{0};
  const char* access_end_addr = addr + access_size;
  for (char* mem_addr = addr; mem_addr < access_end_addr; mem_addr += (1024 * 64)) {
    result = read_64B_accesses<1024>(mem_addr);
  }
  // Returning the last read, not supporting dependent reads
  return result;
}

template <int ACCESS_COUNT_64B>
inline void read_64B_accesses(const std::vector<char*>& addresses) {
  constexpr size_t vector_access_count = SIMD_VECTOR_SIZE_FACTOR * ACCESS_COUNT_64B;
  for (char* addr : addresses) {
    volatile CharVecSIMD* volatile_addr = reinterpret_cast<CharVecSIMD*>(addr);
#pragma GCC unroll 4096
    for (size_t access_idx = 0; access_idx < vector_access_count; ++access_idx) {
      auto result = volatile_addr[access_idx];
    }
  }
}

inline void read_64(const std::vector<char*>& addresses) { read_64B_accesses<1>(addresses); }

inline void read_128(const std::vector<char*>& addresses) { read_64B_accesses<2>(addresses); }

inline void read_256(const std::vector<char*>& addresses) { read_64B_accesses<4>(addresses); }

inline void read_512(const std::vector<char*>& addresses) { read_64B_accesses<8>(addresses); }

inline void read_1k(const std::vector<char*>& addresses) { read_64B_accesses<16>(addresses); }

inline void read_2k(const std::vector<char*>& addresses) { read_64B_accesses<32>(addresses); }

inline void read_4k(const std::vector<char*>& addresses) { read_64B_accesses<64>(addresses); }

inline void read_8k(const std::vector<char*>& addresses) { read_64B_accesses<128>(addresses); }

inline void read_16k(const std::vector<char*>& addresses) { read_64B_accesses<256>(addresses); }

inline void read_32k(const std::vector<char*>& addresses) { read_64B_accesses<512>(addresses); }

inline void read_64k(const std::vector<char*>& addresses) { read_64B_accesses<1024>(addresses); }

inline void read(const std::vector<char*>& addresses, const size_t access_size) {
  for (char* addr : addresses) {
    const char* access_end_addr = addr + access_size;
    // Note that it might make sense to use a modified version of read_64k() here, but that this requires
    // inspecting the assembly instructions again so that we do not introduce overhead that we can avoid.
    for (char* mem_addr = addr; mem_addr < access_end_addr; mem_addr += (1024 * 64)) {
      // Read in 64KiB chunks
      read_64B_accesses<1024>(mem_addr);
    }
  }
}

/**
 * #####################################################
 * WRITE SCALAR OPERATIONS
 * #####################################################
 */

inline void write_data_scalar(char* start_address, const char* end_address) {
  for (char* mem_addr = start_address; mem_addr < end_address; mem_addr += BASE_ACCESS_SIZE) {
    std::memcpy(mem_addr, WRITE_DATA, BASE_ACCESS_SIZE);
  }
}

/**
 * #####################################################
 * BASE STORE OPERATIONS
 * #####################################################
 */

#if defined(__ARM_NEON)
template <int ACCESS_COUNT_64B>
inline void write_64B_accesses(char* address) {
  const auto* data = reinterpret_cast<const uint8x16x4_t*>(WRITE_DATA);
  auto* base_address = reinterpret_cast<uint8_t*>(address);
  constexpr size_t vector_access_count = SIMD_VECTOR_SIZE_FACTOR * ACCESS_COUNT_64B;
#pragma GCC unroll 4096
  for (size_t access_idx = 0; access_idx < vector_access_count; ++access_idx) {
    vst1q_u8_x4(base_address + (SIMD_VECTOR_SIZE * access_idx), *data);
  }
}

#else
template <int ACCESS_COUNT_64B>
inline void write_64B_accesses(char* address) {
  const CharVecSIMD* write_data = reinterpret_cast<const CharVecSIMD*>(WRITE_DATA);
  CharVecSIMD* target_address = reinterpret_cast<CharVecSIMD*>(address);
  constexpr size_t vector_access_count = SIMD_VECTOR_SIZE_FACTOR * ACCESS_COUNT_64B;
#pragma GCC unroll 4096
  for (size_t access_idx = 0; access_idx < vector_access_count; access_idx++) {
    target_address[access_idx] = write_data[0];
  }
}
#endif

template <int ACCESS_COUNT_64B>
inline void write_64B_accesses(char* address, flush_fn flush, barrier_fn barrier) {
  write_64B_accesses<ACCESS_COUNT_64B>(address);
  flush(address, 64 * ACCESS_COUNT_64B);
  barrier();
}

inline void write_data_range(char* start_addr, const char* end_addr) {
  for (char* mem_addr = start_addr; mem_addr < end_addr; mem_addr += 64) {
    write_64B_accesses<1>(mem_addr);
  }
}

inline void write_data(char* start_address, const char* end_address) {
  return write_data_range(start_address, end_address);
}

inline void write_64(char* addr, flush_fn flush, barrier_fn barrier) { write_64B_accesses<1>(addr, flush, barrier); }

inline void write_128(char* addr, flush_fn flush, barrier_fn barrier) { write_64B_accesses<2>(addr, flush, barrier); }

inline void write_256(char* addr, flush_fn flush, barrier_fn barrier) { write_64B_accesses<4>(addr, flush, barrier); }

inline void write_512(char* addr, flush_fn flush, barrier_fn barrier) { write_64B_accesses<8>(addr, flush, barrier); }

inline void write_1k(char* addr, flush_fn flush, barrier_fn barrier) { write_64B_accesses<16>(addr, flush, barrier); }

inline void write_2k(char* addr, flush_fn flush, barrier_fn barrier) { write_64B_accesses<32>(addr, flush, barrier); }

inline void write_4k(char* addr, flush_fn flush, barrier_fn barrier) { write_64B_accesses<64>(addr, flush, barrier); }

inline void write_8k(char* addr, flush_fn flush, barrier_fn barrier) { write_64B_accesses<128>(addr, flush, barrier); }

inline void write_16k(char* addr, flush_fn flush, barrier_fn barrier) { write_64B_accesses<256>(addr, flush, barrier); }

inline void write_32k(char* addr, flush_fn flush, barrier_fn barrier) { write_64B_accesses<512>(addr, flush, barrier); }

inline void write_64k(char* addr, flush_fn flush, barrier_fn barrier) {
  write_64B_accesses<1024>(addr, flush, barrier);
}

inline void write(char* addr, const size_t access_size, flush_fn flush, barrier_fn barrier) {
  const char* access_end_addr = addr + access_size;
  for (char* mem_addr = addr; mem_addr < access_end_addr; mem_addr += (1024 * 64)) {
    write_64B_accesses<1024>(mem_addr);
  }
  flush(addr, access_size);
  barrier();
}

/**
 * #####################################################
 * STORE OPERATIONS (MULTIPLE ADDRESSES)
 * #####################################################
 */

inline void write(const std::vector<char*>& addresses, flush_fn flush, barrier_fn barrier, write_fn write_access) {
  for (auto* addr : addresses) {
    write_access(addr, flush, barrier);
  }
}

inline void write(const std::vector<char*>& addresses, const size_t access_size, flush_fn flush, barrier_fn barrier) {
  for (auto* addr : addresses) {
    // Write in 64KiB chunks
    write(addr, access_size, flush, barrier);
  }
}

/**
 * #####################################################
 * STORE + CLWB OPERATIONS
 * #####################################################
 */

#ifdef HAS_CLWB
inline void write_clwb_64(char* addr) { write_64(addr, flush_clwb, sfence_barrier); }

inline void write_clwb_128(char* addr) { write_128(addr, flush_clwb, sfence_barrier); }

inline void write_clwb_256(char* addr) { write_256(addr, flush_clwb, sfence_barrier); }

inline void write_clwb_512(char* addr) { write_512(addr, flush_clwb, sfence_barrier); }

inline void write_clwb_1k(char* addr) { write_1k(addr, flush_clwb, sfence_barrier); }

inline void write_clwb_2k(char* addr) { write_2k(addr, flush_clwb, sfence_barrier); }

inline void write_clwb_4k(char* addr) { write_4k(addr, flush_clwb, sfence_barrier); }

inline void write_clwb_8k(char* addr) { write_8k(addr, flush_clwb, sfence_barrier); }

inline void write_clwb_16k(char* addr) { write_16k(addr, flush_clwb, sfence_barrier); }

inline void write_clwb_32k(char* addr) { write_32k(addr, flush_clwb, sfence_barrier); }

inline void write_clwb_64k(char* addr) { write_64k(addr, flush_clwb, sfence_barrier); }

inline void write_clwb(char* addr, const size_t access_size) { write(addr, access_size, flush_clwb, sfence_barrier); }

inline void write_clwb_64(const std::vector<char*>& addresses) {
  write(addresses, flush_clwb, sfence_barrier, write_64);
}

inline void write_clwb_128(const std::vector<char*>& addresses) {
  write(addresses, flush_clwb, sfence_barrier, write_128);
}

inline void write_clwb_256(const std::vector<char*>& addresses) {
  write(addresses, flush_clwb, sfence_barrier, write_256);
}

inline void write_clwb_512(const std::vector<char*>& addresses) {
  write(addresses, flush_clwb, sfence_barrier, write_512);
}

inline void write_clwb_1k(const std::vector<char*>& addresses) {
  write(addresses, flush_clwb, sfence_barrier, write_1k);
}

inline void write_clwb_2k(const std::vector<char*>& addresses) {
  write(addresses, flush_clwb, sfence_barrier, write_2k);
}

inline void write_clwb_4k(const std::vector<char*>& addresses) {
  write(addresses, flush_clwb, sfence_barrier, write_4k);
}

inline void write_clwb_8k(const std::vector<char*>& addresses) {
  write(addresses, flush_clwb, sfence_barrier, write_8k);
}

inline void write_clwb_16k(const std::vector<char*>& addresses) {
  write(addresses, flush_clwb, sfence_barrier, write_16k);
}

inline void write_clwb_32k(const std::vector<char*>& addresses) {
  write(addresses, flush_clwb, sfence_barrier, write_32k);
}

inline void write_clwb_64k(const std::vector<char*>& addresses) {
  write(addresses, flush_clwb, sfence_barrier, write_64k);
}

inline void write_clwb(const std::vector<char*>& addresses, const size_t access_size) {
  write(addresses, access_size, flush_clwb, sfence_barrier);
}

#endif  // clwb

/**
 * #####################################################
 * STORE-ONLY OPERATIONS
 * #####################################################
 */

inline void write_none_64(char* addr) { write_64(addr, no_flush, no_barrier); }

inline void write_none_128(char* addr) { write_128(addr, no_flush, no_barrier); }

inline void write_none_256(char* addr) { write_256(addr, no_flush, no_barrier); }

inline void write_none_512(char* addr) { write_512(addr, no_flush, no_barrier); }

inline void write_none_1k(char* addr) { write_1k(addr, no_flush, no_barrier); }

inline void write_none_2k(char* addr) { write_2k(addr, no_flush, no_barrier); }

inline void write_none_4k(char* addr) { write_4k(addr, no_flush, no_barrier); }

inline void write_none_8k(char* addr) { write_8k(addr, no_flush, no_barrier); }

inline void write_none_16k(char* addr) { write_16k(addr, no_flush, no_barrier); }

inline void write_none_32k(char* addr) { write_32k(addr, no_flush, no_barrier); }

inline void write_none_64k(char* addr) { write_64k(addr, no_flush, no_barrier); }

inline void write_none(char* addr, const size_t access_size) { write(addr, access_size, no_flush, no_barrier); }

inline void write_none_64(const std::vector<char*>& addresses) { write(addresses, no_flush, no_barrier, write_64); }

inline void write_none_128(const std::vector<char*>& addresses) { write(addresses, no_flush, no_barrier, write_128); }

inline void write_none_256(const std::vector<char*>& addresses) { write(addresses, no_flush, no_barrier, write_256); }

inline void write_none_512(const std::vector<char*>& addresses) { write(addresses, no_flush, no_barrier, write_512); }

inline void write_none_1k(const std::vector<char*>& addresses) { write(addresses, no_flush, no_barrier, write_1k); }

inline void write_none_2k(const std::vector<char*>& addresses) { write(addresses, no_flush, no_barrier, write_2k); }

inline void write_none_4k(const std::vector<char*>& addresses) { write(addresses, no_flush, no_barrier, write_4k); }

inline void write_none_8k(const std::vector<char*>& addresses) { write(addresses, no_flush, no_barrier, write_8k); }

inline void write_none_16k(const std::vector<char*>& addresses) { write(addresses, no_flush, no_barrier, write_16k); }

inline void write_none_32k(const std::vector<char*>& addresses) { write(addresses, no_flush, no_barrier, write_32k); }

inline void write_none_64k(const std::vector<char*>& addresses) { write(addresses, no_flush, no_barrier, write_64k); }

inline void write_none(const std::vector<char*>& addresses, const size_t access_size) {
  write(addresses, access_size, no_flush, no_barrier);
}

#if defined(USE_AVX_2) || defined(USE_AVX_512)
/**
 * #####################################################
 * NON_TEMPORAL STORE OPERATIONS
 * #####################################################
 */

inline void simd_write_nt_64(char* addr) { simd_write_nt_64B_accesses_sfence<1>(addr); }

inline void simd_write_nt_128(char* addr) { simd_write_nt_64B_accesses_sfence<2>(addr); }

inline void simd_write_nt_256(char* addr) { simd_write_nt_64B_accesses_sfence<4>(addr); }

inline void simd_write_nt_512(char* addr) { simd_write_nt_64B_accesses_sfence<8>(addr); }

inline void simd_write_nt_1k(char* addr) { simd_write_nt_64B_accesses_sfence<16>(addr); }

inline void simd_write_nt_2k(char* addr) { simd_write_nt_64B_accesses_sfence<32>(addr); }

inline void simd_write_nt_4k(char* addr) { simd_write_nt_64B_accesses_sfence<64>(addr); }

inline void simd_write_nt_8k(char* addr) { simd_write_nt_64B_accesses_sfence<128>(addr); }

inline void simd_write_nt_16k(char* addr) { simd_write_nt_64B_accesses_sfence<256>(addr); }

inline void simd_write_nt_32k(char* addr) { simd_write_nt_64B_accesses_sfence<512>(addr); }

inline void simd_write_nt_64k(char* addr) { simd_write_nt_64B_accesses_sfence<1024>(addr); }

inline void simd_write_nt(char* addr, const size_t access_size) {
  const char* access_end_addr = addr + access_size;
  for (char* mem_addr = addr; mem_addr < access_end_addr; mem_addr += (1024 * 64)) {
    simd_write_nt_64B_accesses<1024>(addr);
  }
  sfence_barrier();
}

/**
 * #####################################################
 * NON_TEMPORAL STORE OPERATIONS (MULTIPLE ADDRESSES)
 * #####################################################
 */
inline void simd_write_nt(const std::vector<char*>& addresses, simd_write_nt_fn simd_write_nt_access) {
  for (auto* addr : addresses) {
    simd_write_nt_access(addr);
  }
}

inline void simd_write_nt_64(const std::vector<char*>& addresses) { simd_write_nt(addresses, simd_write_nt_64); }

inline void simd_write_nt_128(const std::vector<char*>& addresses) { simd_write_nt(addresses, simd_write_nt_128); }

inline void simd_write_nt_256(const std::vector<char*>& addresses) { simd_write_nt(addresses, simd_write_nt_256); }

inline void simd_write_nt_512(const std::vector<char*>& addresses) { simd_write_nt(addresses, simd_write_nt_512); }

inline void simd_write_nt_1k(const std::vector<char*>& addresses) { simd_write_nt(addresses, simd_write_nt_1k); }

inline void simd_write_nt_2k(const std::vector<char*>& addresses) { simd_write_nt(addresses, simd_write_nt_2k); }

inline void simd_write_nt_4k(const std::vector<char*>& addresses) { simd_write_nt(addresses, simd_write_nt_4k); }

inline void simd_write_nt_8k(const std::vector<char*>& addresses) { simd_write_nt(addresses, simd_write_nt_8k); }

inline void simd_write_nt_16k(const std::vector<char*>& addresses) { simd_write_nt(addresses, simd_write_nt_16k); }

inline void simd_write_nt_32k(const std::vector<char*>& addresses) { simd_write_nt(addresses, simd_write_nt_32k); }

inline void simd_write_nt_64k(const std::vector<char*>& addresses) { simd_write_nt(addresses, simd_write_nt_64k); }

inline void simd_write_nt(const std::vector<char*>& addresses, const size_t access_size) {
  for (auto* addr : addresses) {
    // Write in 64KiB chunks
    simd_write_nt(addr, access_size);
  }
}

#endif  // defined(USE_AVX_2) || defined(USE_AVX_512)
}  // namespace mema::rw_ops
