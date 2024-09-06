#pragma once

#include <cstddef>

#if defined(HAS_CLWB) || defined(USE_AVX_2) || defined(USE_AVX_512)
#include <immintrin.h>
#endif

namespace cxlbench::rw_ops {

#if defined(__powerpc__)
static constexpr auto CACHE_LINE_SIZE = size_t{128};

#else
static constexpr auto CACHE_LINE_SIZE = size_t{64};
#endif

// Write data for a base access (e.g., 64 B access). We assume vector sizes of 16/32/64 Bytes. We repeat the write data
// after 16 characters for a simpler write logic with vector sizes of 16/32 Bytes.
static const char WRITE_DATA[] __attribute__((aligned(64))) =
    "AbcdefghijklmnopAbcdefghijklmnopAbcdefghijklmnopAbcdefghijklmnop";

// We use platform-independent vector intrinsics (compiler intrinsics) to avoid writing platform-specific SIMD
// intrinsics (only for char data type; see definition of CharVec). SIMD_VECTOR_SIZE defines the vector size and, in the
// case of type char, the number of Bytes being accessed. 64 B is the base access size in this benchmark tool. We need a
// certain number of vectorized memory accesses to access 64 B. This number is determined by SIMD_VECTOR_SIZE_FACTOR.

#if defined(__powerpc__)
// 128 bit registers
static constexpr size_t SIMD_VECTOR_SIZE = 16;
#elif defined(USE_AVX_2)
// 256 bit registers
static constexpr size_t SIMD_VECTOR_SIZE = 32;
#else
// 512 bit registers
static constexpr size_t SIMD_VECTOR_SIZE = 64;
#endif

static constexpr size_t BASE_ACCESS_SIZE = 64;
// This factor needs to be multiplied with the a SIMD instruction's access size to achieve the base access size.
static constexpr size_t SIMD_VECTOR_SIZE_FACTOR = BASE_ACCESS_SIZE / SIMD_VECTOR_SIZE;

using CharVecBase __attribute__((vector_size(BASE_ACCESS_SIZE))) = char;
using CharVecSIMD __attribute__((vector_size(SIMD_VECTOR_SIZE))) = char;
using CharVec16 __attribute__((vector_size(16))) = char;
using CharVec32 __attribute__((vector_size(32))) = char;

using cache_func = void(char*, const size_t);
using barrier_func = void();
using write_func = void(char*, cache_func, barrier_func);
using write_stream_func = void(char*);

/** no explicit cache instruction is used. */
inline void no_cache_fn(char* addr, const size_t len) {}

/** no memory order is guaranteed. */
inline void no_barrier() {}

#if defined(HAS_CLWB) || defined(USE_AVX_2) || defined(USE_AVX_512)
/** Use sfence to guarantee memory order on x86. Earlier store operations cannot be reordered beyond this point. */
inline void sfence_barrier() { _mm_sfence(); }
inline void lfence_barrier() { _mm_lfence(); }
inline void mfence_barrier() { _mm_mfence(); }
inline u64 cycles_now() { return __rdtsc(); }
#endif

/** flush the cache line using clwb. */
#ifdef HAS_CLWB
inline void cache_clwb(char* addr, const size_t len) {
  const auto* end_addr = addr + len;
  for (auto* current_cl = addr; current_cl < end_addr; current_cl += CACHE_LINE_SIZE) {
    _mm_clwb(current_cl);
  }
}
#endif

/** flush the cache line using clflush. */
#ifdef HAS_CLFLUSH
inline void cache_clflush(char* addr, const size_t len) {
  const char* end_addr = addr + len;
  for (char* current_cl = addr; current_cl < end_addr; current_cl += CACHE_LINE_SIZE) {
    _mm_clflush(current_cl);
  }
}
#endif

/** flush the cache line using clflushopt. */
#ifdef HAS_CLFLUSHOPT
inline void cache_clflushopt(char* addr, const size_t len) {
  const char* end_addr = addr + len;
  for (char* current_cl = addr; current_cl < end_addr; current_cl += CACHE_LINE_SIZE) {
    _mm_clflushopt(current_cl);
  }
}
#endif

inline void x100_nop() {
  asm volatile(
      "nop \nnop \nnop \nnop \nnop \nnop \nnop \nnop \nnop \nnop \n"
      "nop \nnop \nnop \nnop \nnop \nnop \nnop \nnop \nnop \nnop \n"
      "nop \nnop \nnop \nnop \nnop \nnop \nnop \nnop \nnop \nnop \n"
      "nop \nnop \nnop \nnop \nnop \nnop \nnop \nnop \nnop \nnop \n"
      "nop \nnop \nnop \nnop \nnop \nnop \nnop \nnop \nnop \nnop \n"
      "nop \nnop \nnop \nnop \nnop \nnop \nnop \nnop \nnop \nnop \n"
      "nop \nnop \nnop \nnop \nnop \nnop \nnop \nnop \nnop \nnop \n"
      "nop \nnop \nnop \nnop \nnop \nnop \nnop \nnop \nnop \nnop \n"
      "nop \nnop \nnop \nnop \nnop \nnop \nnop \nnop \nnop \nnop \n"
      "nop \nnop \nnop \nnop \nnop \nnop \nnop \nnop \nnop \nnop \n");
}

}  // namespace cxlbench::rw_ops
