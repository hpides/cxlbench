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
#include "types.hpp"

/** Neon intrinsics for ARM */
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace cxlbench::rw_ops {

/**
 * #####################################################
 * READ OPERATIONS
 * #####################################################
 */

template <int ACCESS_COUNT_64B>
inline CharVecBase read_64B_accesses(char* address, cache_func cache_fn, barrier_func barrier) {
  volatile CharVecSIMD* volatile_addr = reinterpret_cast<CharVecSIMD*>(address);
  auto result = CharVecBase{0};
  auto result_vec_simd = reinterpret_cast<CharVecSIMD*>(&result);
  // The maximum access size is 64 KiB. With a 64 B base access size, we need 1024 accesses.
#pragma GCC unroll 1024
  for (auto base_access_idx = u64{0}; base_access_idx < ACCESS_COUNT_64B; ++base_access_idx) {
    cache_fn(address + (base_access_idx * 64), 64);
    barrier();
// Perform base access with SIMD_VECTOR_SIZE_FACTOR * vector accesses.
#pragma GCC unroll SIMD_VECTOR_SIZE_FACTOR
    for (auto sub_access_idx = u64{0}; sub_access_idx < SIMD_VECTOR_SIZE_FACTOR; ++sub_access_idx) {
      const auto index = (base_access_idx * SIMD_VECTOR_SIZE_FACTOR) + sub_access_idx;
      result_vec_simd[index] = volatile_addr[index];
    }
  }
  return result;
}

inline u64 read_8_get_u64(char* addr) {
  volatile u64* volatile_addr = reinterpret_cast<u64*>(addr);
  return volatile_addr[0];
}

inline char read_4(char* addr, cache_func cache_fn, barrier_func barrier) {
  cache_fn(addr, 64);
  barrier();
  volatile u32* volatile_addr = reinterpret_cast<u32*>(addr);
  auto result = volatile_addr[0];
  return static_cast<char>(result);
}

inline char read_8(char* addr, cache_func cache_fn, barrier_func barrier) {
  cache_fn(addr, 64);
  barrier();
  volatile u64* volatile_addr = reinterpret_cast<u64*>(addr);
  auto result = volatile_addr[0];
  return static_cast<char>(result);
}

inline char read_16(char* addr, cache_func cache_fn, barrier_func barrier) {
  cache_fn(addr, 64);
  barrier();
  volatile CharVec16* volatile_addr = reinterpret_cast<CharVec16*>(addr);
  auto result = volatile_addr[0];
  return static_cast<char>(result[0]);
}

inline char read_32(char* addr, cache_func cache_fn, barrier_func barrier) {
  cache_fn(addr, 64);
  barrier();
  volatile CharVec32* volatile_addr = reinterpret_cast<CharVec32*>(addr);
  auto result = volatile_addr[0];
  return static_cast<char>(result[0]);
}

inline char read_64(char* addr, cache_func cache_fn, barrier_func barrier) {
  return static_cast<char>(read_64B_accesses<1>(addr, cache_fn, barrier)[0]);
}

inline u64 read_64_get_u64(char* addr) {
  volatile CharVecSIMD* volatile_addr = reinterpret_cast<CharVecSIMD*>(addr);
  // Assuming vector size 64
  auto res = volatile_addr[0];
  const auto* u64_res = reinterpret_cast<const u64*>(&res);
  return u64_res[0];
}

inline char read_128(char* addr, cache_func cache_fn, barrier_func barrier) {
  return static_cast<char>(read_64B_accesses<2>(addr, cache_fn, barrier)[0]);
}

inline char read_256(char* addr, cache_func cache_fn, barrier_func barrier) {
  return static_cast<char>(read_64B_accesses<4>(addr, cache_fn, barrier)[0]);
}

inline char read_512(char* addr, cache_func cache_fn, barrier_func barrier) {
  return static_cast<char>(read_64B_accesses<8>(addr, cache_fn, barrier)[0]);
}

inline char read_1k(char* addr, cache_func cache_fn, barrier_func barrier) {
  return static_cast<char>(read_64B_accesses<16>(addr, cache_fn, barrier)[0]);
}

inline char read_2k(char* addr, cache_func cache_fn, barrier_func barrier) {
  return static_cast<char>(read_64B_accesses<32>(addr, cache_fn, barrier)[0]);
}

inline char read_4k(char* addr, cache_func cache_fn, barrier_func barrier) {
  return static_cast<char>(read_64B_accesses<64>(addr, cache_fn, barrier)[0]);
}

inline char read_8k(char* addr, cache_func cache_fn, barrier_func barrier) {
  return static_cast<char>(read_64B_accesses<128>(addr, cache_fn, barrier)[0]);
}

inline char read_16k(char* addr, cache_func cache_fn, barrier_func barrier) {
  return static_cast<char>(read_64B_accesses<256>(addr, cache_fn, barrier)[0]);
}

inline char read_32k(char* addr, cache_func cache_fn, barrier_func barrier) {
  return static_cast<char>(read_64B_accesses<512>(addr, cache_fn, barrier)[0]);
}

inline char read_64k(char* addr, cache_func cache_fn, barrier_func barrier) {
  return static_cast<char>(read_64B_accesses<1024>(addr, cache_fn, barrier)[0]);
}

inline char read(char* addr, const size_t access_size, cache_func cache_fn, barrier_func barrier) {
  auto result = CharVecBase{0};
  const char* access_end_addr = addr + access_size;
  for (char* mem_addr = addr; mem_addr < access_end_addr; mem_addr += (1024 * 64)) {
    result = read_64B_accesses<1024>(mem_addr, cache_fn, barrier);
  }
  // Returning the last read, not supporting dependent reads
  return static_cast<char>(result[0]);
}

inline char read_none_8(char* addr) { return read_8(addr, no_cache_fn, no_barrier); }

// TODO(MW) add all sizes.
inline char read_none_64(char* addr) { return read_64(addr, no_cache_fn, no_barrier); }

/**
 * #####################################################
 * LOAD OPERATIONS (MULTIPLE ADDRESSES)
 * #####################################################
 */

template <int ACCESS_COUNT_64B>
inline void read_64B_accesses(const std::vector<char*>& addresses, cache_func cache_fn, barrier_func barrier) {
  constexpr size_t vector_access_count = SIMD_VECTOR_SIZE_FACTOR * ACCESS_COUNT_64B;
  for (char* addr : addresses) {
    cache_fn(addr, 64);
    barrier();
    volatile CharVecSIMD* volatile_addr = reinterpret_cast<CharVecSIMD*>(addr);
#pragma GCC unroll 4096
    for (size_t access_idx = 0; access_idx < vector_access_count; ++access_idx) {
      auto result = volatile_addr[access_idx];
    }
  }
}

inline void read_4(const std::vector<char*>& addresses, cache_func cache_fn, barrier_func barrier) {
  for (char* addr : addresses) {
    cache_fn(addr, 64);
    barrier();
    volatile u32* volatile_addr = reinterpret_cast<u32*>(addr);
    auto result = volatile_addr[0];
  }
}

inline void read_8(const std::vector<char*>& addresses, cache_func cache_fn, barrier_func barrier) {
  for (char* addr : addresses) {
    cache_fn(addr, 64);
    barrier();
    volatile u64* volatile_addr = reinterpret_cast<u64*>(addr);
    auto result = volatile_addr[0];
  }
}

inline void read_16(const std::vector<char*>& addresses, cache_func cache_fn, barrier_func barrier) {
  for (char* addr : addresses) {
    cache_fn(addr, 64);
    barrier();
    volatile CharVec16* volatile_addr = reinterpret_cast<CharVec16*>(addr);
    auto result = volatile_addr[0];
  }
}

inline void read_32(const std::vector<char*>& addresses, cache_func cache_fn, barrier_func barrier) {
  for (char* addr : addresses) {
    cache_fn(addr, 64);
    barrier();
    volatile CharVec32* volatile_addr = reinterpret_cast<CharVec32*>(addr);
    auto result = volatile_addr[0];
  }
}

inline void read_64(const std::vector<char*>& addresses, cache_func cache_fn, barrier_func barrier) {
  read_64B_accesses<1>(addresses, cache_fn, barrier);
}

inline void read_128(const std::vector<char*>& addresses, cache_func cache_fn, barrier_func barrier) {
  read_64B_accesses<2>(addresses, cache_fn, barrier);
}

inline void read_256(const std::vector<char*>& addresses, cache_func cache_fn, barrier_func barrier) {
  read_64B_accesses<4>(addresses, cache_fn, barrier);
}

inline void read_512(const std::vector<char*>& addresses, cache_func cache_fn, barrier_func barrier) {
  read_64B_accesses<8>(addresses, cache_fn, barrier);
}

inline void read_1k(const std::vector<char*>& addresses, cache_func cache_fn, barrier_func barrier) {
  read_64B_accesses<16>(addresses, cache_fn, barrier);
}

inline void read_2k(const std::vector<char*>& addresses, cache_func cache_fn, barrier_func barrier) {
  read_64B_accesses<32>(addresses, cache_fn, barrier);
}

inline void read_4k(const std::vector<char*>& addresses, cache_func cache_fn, barrier_func barrier) {
  read_64B_accesses<64>(addresses, cache_fn, barrier);
}

inline void read_8k(const std::vector<char*>& addresses, cache_func cache_fn, barrier_func barrier) {
  read_64B_accesses<128>(addresses, cache_fn, barrier);
}

inline void read_16k(const std::vector<char*>& addresses, cache_func cache_fn, barrier_func barrier) {
  read_64B_accesses<256>(addresses, cache_fn, barrier);
}

inline void read_32k(const std::vector<char*>& addresses, cache_func cache_fn, barrier_func barrier) {
  read_64B_accesses<512>(addresses, cache_fn, barrier);
}

inline void read_64k(const std::vector<char*>& addresses, cache_func cache_fn, barrier_func barrier) {
  read_64B_accesses<1024>(addresses, cache_fn, barrier);
}

inline void read(const std::vector<char*>& addresses, const size_t access_size, cache_func cache_fn,
                 barrier_func barrier) {
  for (char* addr : addresses) {
    const char* access_end_addr = addr + access_size;
    // Note that it might make sense to use a modified version of read_64k() here, but that this requires
    // inspecting the assembly instructions again so that we do not introduce overhead that we can avoid.
    for (char* mem_addr = addr; mem_addr < access_end_addr; mem_addr += (1024 * 64)) {
      // Read in 64KiB batches
      read_64B_accesses<1024>(mem_addr, cache_fn, barrier);
    }
  }
}

inline void read_none_8(const std::vector<char*>& addresses) { read_8(addresses, no_cache_fn, no_barrier); }

// TODO(MW) add other sizes
inline void read_none_64(const std::vector<char*>& addresses) { read_64(addresses, no_cache_fn, no_barrier); }

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
  auto* base_address = reinterpret_cast<u8*>(address);
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
inline void write_64B_accesses(char* address, cache_func cache_fn, barrier_func barrier) {
  write_64B_accesses<ACCESS_COUNT_64B>(address);
  cache_fn(address, 64 * ACCESS_COUNT_64B);
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

inline void write_4(char* addr, cache_func cache_fn, barrier_func barrier) {
  const u32* write_data = reinterpret_cast<const u32*>(WRITE_DATA);
  u32* target_address = reinterpret_cast<u32*>(addr);
  target_address[0] = write_data[0];
  cache_fn(addr, 4);
  barrier();
}

inline void write_8(char* addr, cache_func cache_fn, barrier_func barrier) {
  const u64* write_data = reinterpret_cast<const u64*>(WRITE_DATA);
  u64* target_address = reinterpret_cast<u64*>(addr);
  target_address[0] = write_data[0];
  cache_fn(addr, 8);
  barrier();
}

inline void write_16(char* addr, cache_func cache_fn, barrier_func barrier) {
  const CharVec16* write_data = reinterpret_cast<const CharVec16*>(WRITE_DATA);
  CharVec16* target_address = reinterpret_cast<CharVec16*>(addr);
  target_address[0] = write_data[0];
  cache_fn(addr, 16);
  barrier();
}

inline void write_32(char* addr, cache_func cache_fn, barrier_func barrier) {
  const CharVec32* write_data = reinterpret_cast<const CharVec32*>(WRITE_DATA);
  CharVec32* target_address = reinterpret_cast<CharVec32*>(addr);
  target_address[0] = write_data[0];
  cache_fn(addr, 32);
  barrier();
}

inline void write_64(char* addr, cache_func cache_fn, barrier_func barrier) {
  write_64B_accesses<1>(addr, cache_fn, barrier);
}

inline void write_128(char* addr, cache_func cache_fn, barrier_func barrier) {
  write_64B_accesses<2>(addr, cache_fn, barrier);
}

inline void write_256(char* addr, cache_func cache_fn, barrier_func barrier) {
  write_64B_accesses<4>(addr, cache_fn, barrier);
}

inline void write_512(char* addr, cache_func cache_fn, barrier_func barrier) {
  write_64B_accesses<8>(addr, cache_fn, barrier);
}

inline void write_1k(char* addr, cache_func cache_fn, barrier_func barrier) {
  write_64B_accesses<16>(addr, cache_fn, barrier);
}

inline void write_2k(char* addr, cache_func cache_fn, barrier_func barrier) {
  write_64B_accesses<32>(addr, cache_fn, barrier);
}

inline void write_4k(char* addr, cache_func cache_fn, barrier_func barrier) {
  write_64B_accesses<64>(addr, cache_fn, barrier);
}

inline void write_8k(char* addr, cache_func cache_fn, barrier_func barrier) {
  write_64B_accesses<128>(addr, cache_fn, barrier);
}

inline void write_16k(char* addr, cache_func cache_fn, barrier_func barrier) {
  write_64B_accesses<256>(addr, cache_fn, barrier);
}

inline void write_32k(char* addr, cache_func cache_fn, barrier_func barrier) {
  write_64B_accesses<512>(addr, cache_fn, barrier);
}

inline void write_64k(char* addr, cache_func cache_fn, barrier_func barrier) {
  write_64B_accesses<1024>(addr, cache_fn, barrier);
}

inline void write(char* addr, const size_t access_size, cache_func cache_fn, barrier_func barrier) {
  const char* access_end_addr = addr + access_size;
  for (char* mem_addr = addr; mem_addr < access_end_addr; mem_addr += (1024 * 64)) {
    write_64B_accesses<1024>(mem_addr);
  }
  cache_fn(addr, access_size);
  barrier();
}

/**
 * #####################################################
 * STORE OPERATIONS (MULTIPLE ADDRESSES)
 * #####################################################
 */

inline void write(const std::vector<char*>& addresses, cache_func cache_fn, barrier_func barrier,
                  write_func write_access) {
  for (auto* addr : addresses) {
    write_access(addr, cache_fn, barrier);
  }
}

inline void write(const std::vector<char*>& addresses, const size_t access_size, cache_func cache_fn,
                  barrier_func barrier) {
  for (auto* addr : addresses) {
    // Write in 64KiB batches
    write(addr, access_size, cache_fn, barrier);
  }
}

#ifdef HAS_CLFLUSHOPT
inline void read_flushopt_8(const std::vector<char*>& addresses) {
  read_8(addresses, cache_clflushopt, sfence_barrier);
}

inline void read_flushopt_64(const std::vector<char*>& addresses) {
  read_64(addresses, cache_clflushopt, sfence_barrier);
}
#endif  // clflushopt

#ifdef HAS_CLFLUSH
inline void read_flush_8(const std::vector<char*>& addresses) { read_8(addresses, cache_clflush, no_barrier); }

inline void read_flush_64(const std::vector<char*>& addresses) { read_64(addresses, cache_clflush, no_barrier); }
#endif  // clflush

#ifdef HAS_CLWB
inline void write_clwb_8(char* addr) { write_8(addr, cache_clwb, sfence_barrier); }

inline void write_clwb_64(char* addr) { write_64(addr, cache_clwb, sfence_barrier); }

inline void write_clwb_128(char* addr) { write_128(addr, cache_clwb, sfence_barrier); }

inline void write_clwb_256(char* addr) { write_256(addr, cache_clwb, sfence_barrier); }

inline void write_clwb_512(char* addr) { write_512(addr, cache_clwb, sfence_barrier); }

inline void write_clwb_1k(char* addr) { write_1k(addr, cache_clwb, sfence_barrier); }

inline void write_clwb_2k(char* addr) { write_2k(addr, cache_clwb, sfence_barrier); }

inline void write_clwb_4k(char* addr) { write_4k(addr, cache_clwb, sfence_barrier); }

inline void write_clwb_8k(char* addr) { write_8k(addr, cache_clwb, sfence_barrier); }

inline void write_clwb_16k(char* addr) { write_16k(addr, cache_clwb, sfence_barrier); }

inline void write_clwb_32k(char* addr) { write_32k(addr, cache_clwb, sfence_barrier); }

inline void write_clwb_64k(char* addr) { write_64k(addr, cache_clwb, sfence_barrier); }

inline void write_clwb(char* addr, const size_t access_size) { write(addr, access_size, cache_clwb, sfence_barrier); }

inline void write_clwb_8(const std::vector<char*>& addresses) { write(addresses, cache_clwb, sfence_barrier, write_8); }

inline void write_clwb_64(const std::vector<char*>& addresses) {
  write(addresses, cache_clwb, sfence_barrier, write_64);
}

inline void write_clwb_128(const std::vector<char*>& addresses) {
  write(addresses, cache_clwb, sfence_barrier, write_128);
}

inline void write_clwb_256(const std::vector<char*>& addresses) {
  write(addresses, cache_clwb, sfence_barrier, write_256);
}

inline void write_clwb_512(const std::vector<char*>& addresses) {
  write(addresses, cache_clwb, sfence_barrier, write_512);
}

inline void write_clwb_1k(const std::vector<char*>& addresses) {
  write(addresses, cache_clwb, sfence_barrier, write_1k);
}

inline void write_clwb_2k(const std::vector<char*>& addresses) {
  write(addresses, cache_clwb, sfence_barrier, write_2k);
}

inline void write_clwb_4k(const std::vector<char*>& addresses) {
  write(addresses, cache_clwb, sfence_barrier, write_4k);
}

inline void write_clwb_8k(const std::vector<char*>& addresses) {
  write(addresses, cache_clwb, sfence_barrier, write_8k);
}

inline void write_clwb_16k(const std::vector<char*>& addresses) {
  write(addresses, cache_clwb, sfence_barrier, write_16k);
}

inline void write_clwb_32k(const std::vector<char*>& addresses) {
  write(addresses, cache_clwb, sfence_barrier, write_32k);
}

inline void write_clwb_64k(const std::vector<char*>& addresses) {
  write(addresses, cache_clwb, sfence_barrier, write_64k);
}

inline void write_clwb(const std::vector<char*>& addresses, const size_t access_size) {
  write(addresses, access_size, cache_clwb, sfence_barrier);
}

#endif  // clwb

/**
 * #####################################################
 * STORE-ONLY OPERATIONS
 * #####################################################
 */

inline void write_none_4(char* addr) { write_4(addr, no_cache_fn, no_barrier); }

inline void write_none_8(char* addr) { write_8(addr, no_cache_fn, no_barrier); }

inline void write_none_16(char* addr) { write_8(addr, no_cache_fn, no_barrier); }

inline void write_none_32(char* addr) { write_8(addr, no_cache_fn, no_barrier); }

inline void write_none_64(char* addr) { write_64(addr, no_cache_fn, no_barrier); }

inline void write_none_128(char* addr) { write_128(addr, no_cache_fn, no_barrier); }

inline void write_none_256(char* addr) { write_256(addr, no_cache_fn, no_barrier); }

inline void write_none_512(char* addr) { write_512(addr, no_cache_fn, no_barrier); }

inline void write_none_1k(char* addr) { write_1k(addr, no_cache_fn, no_barrier); }

inline void write_none_2k(char* addr) { write_2k(addr, no_cache_fn, no_barrier); }

inline void write_none_4k(char* addr) { write_4k(addr, no_cache_fn, no_barrier); }

inline void write_none_8k(char* addr) { write_8k(addr, no_cache_fn, no_barrier); }

inline void write_none_16k(char* addr) { write_16k(addr, no_cache_fn, no_barrier); }

inline void write_none_32k(char* addr) { write_32k(addr, no_cache_fn, no_barrier); }

inline void write_none_64k(char* addr) { write_64k(addr, no_cache_fn, no_barrier); }

inline void write_none(char* addr, const size_t access_size) { write(addr, access_size, no_cache_fn, no_barrier); }

inline void write_none_4(const std::vector<char*>& addresses) { write(addresses, no_cache_fn, no_barrier, write_4); }

inline void write_none_8(const std::vector<char*>& addresses) { write(addresses, no_cache_fn, no_barrier, write_8); }

inline void write_none_16(const std::vector<char*>& addresses) { write(addresses, no_cache_fn, no_barrier, write_16); }

inline void write_none_32(const std::vector<char*>& addresses) { write(addresses, no_cache_fn, no_barrier, write_32); }

inline void write_none_64(const std::vector<char*>& addresses) { write(addresses, no_cache_fn, no_barrier, write_64); }

inline void write_none_128(const std::vector<char*>& addresses) {
  write(addresses, no_cache_fn, no_barrier, write_128);
}

inline void write_none_256(const std::vector<char*>& addresses) {
  write(addresses, no_cache_fn, no_barrier, write_256);
}

inline void write_none_512(const std::vector<char*>& addresses) {
  write(addresses, no_cache_fn, no_barrier, write_512);
}

inline void write_none_1k(const std::vector<char*>& addresses) { write(addresses, no_cache_fn, no_barrier, write_1k); }

inline void write_none_2k(const std::vector<char*>& addresses) { write(addresses, no_cache_fn, no_barrier, write_2k); }

inline void write_none_4k(const std::vector<char*>& addresses) { write(addresses, no_cache_fn, no_barrier, write_4k); }

inline void write_none_8k(const std::vector<char*>& addresses) { write(addresses, no_cache_fn, no_barrier, write_8k); }

inline void write_none_16k(const std::vector<char*>& addresses) {
  write(addresses, no_cache_fn, no_barrier, write_16k);
}

inline void write_none_32k(const std::vector<char*>& addresses) {
  write(addresses, no_cache_fn, no_barrier, write_32k);
}

inline void write_none_64k(const std::vector<char*>& addresses) {
  write(addresses, no_cache_fn, no_barrier, write_64k);
}

inline void write_none(const std::vector<char*>& addresses, const size_t access_size) {
  write(addresses, access_size, no_cache_fn, no_barrier);
}

#if defined(USE_AVX_2) || defined(USE_AVX_512)
/**
 * #####################################################
 * NON_TEMPORAL LOAD OPERATIONS
 * #####################################################
 */

inline char read_stream_64(char* addr) { return read_stream_64B_access(addr); }

/**
 * #####################################################
 * NON_TEMPORAL LOAD OPERATIONS (MULTIPLE ADDRESSES)
 * #####################################################
 */

inline void read_stream_64(const std::vector<char*>& addresses) {
  for (auto* addr : addresses) {
    read_stream_64B_access(addr);
  }
}

/**
 * #####################################################
 * NON_TEMPORAL STORE OPERATIONS
 * #####################################################
 */

inline void write_stream_64(char* addr) { write_stream_64B_accesses<1>(addr); }

inline void write_stream_128(char* addr) { write_stream_64B_accesses<2>(addr); }

inline void write_stream_256(char* addr) { write_stream_64B_accesses<4>(addr); }

inline void write_stream_512(char* addr) { write_stream_64B_accesses<8>(addr); }

inline void write_stream_1k(char* addr) { write_stream_64B_accesses<16>(addr); }

inline void write_stream_2k(char* addr) { write_stream_64B_accesses<32>(addr); }

inline void write_stream_4k(char* addr) { write_stream_64B_accesses<64>(addr); }

inline void write_stream_8k(char* addr) { write_stream_64B_accesses<128>(addr); }

inline void write_stream_16k(char* addr) { write_stream_64B_accesses<256>(addr); }

inline void write_stream_32k(char* addr) { write_stream_64B_accesses<512>(addr); }

inline void write_stream_64k(char* addr) { write_stream_64B_accesses<1024>(addr); }

/**
 * #####################################################
 * NON_TEMPORAL STORE OPERATIONS (MULTIPLE ADDRESSES)
 * #####################################################
 */
inline void write_stream(const std::vector<char*>& addresses, write_stream_func write_stream_access) {
  for (auto* addr : addresses) {
    write_stream_access(addr);
  }
}

inline void write_stream_64(const std::vector<char*>& addresses) { write_stream(addresses, write_stream_64); }

inline void write_stream_128(const std::vector<char*>& addresses) { write_stream(addresses, write_stream_128); }

inline void write_stream_256(const std::vector<char*>& addresses) { write_stream(addresses, write_stream_256); }

inline void write_stream_512(const std::vector<char*>& addresses) { write_stream(addresses, write_stream_512); }

inline void write_stream_1k(const std::vector<char*>& addresses) { write_stream(addresses, write_stream_1k); }

inline void write_stream_2k(const std::vector<char*>& addresses) { write_stream(addresses, write_stream_2k); }

inline void write_stream_4k(const std::vector<char*>& addresses) { write_stream(addresses, write_stream_4k); }

inline void write_stream_8k(const std::vector<char*>& addresses) { write_stream(addresses, write_stream_8k); }

inline void write_stream_16k(const std::vector<char*>& addresses) { write_stream(addresses, write_stream_16k); }

inline void write_stream_32k(const std::vector<char*>& addresses) { write_stream(addresses, write_stream_32k); }

inline void write_stream_64k(const std::vector<char*>& addresses) { write_stream(addresses, write_stream_64k); }

#endif  // defined(USE_AVX_2) || defined(USE_AVX_512)
}  // namespace cxlbench::rw_ops
