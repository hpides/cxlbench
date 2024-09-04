#pragma once

#include <cstddef>

#if defined(HAS_CLWB) || defined(USE_AVX_2) || defined(USE_AVX_512)
#include <immintrin.h>
#endif

namespace mema::rw_ops {

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

using flush_fn = void(char*, const size_t);
using barrier_fn = void();
using write_fn = void(char*, flush_fn, barrier_fn);
using simd_write_nt_fn = void(char*);

/** no explicit cache line flush is used. */
inline void no_flush(char* addr, const size_t len) {}

#if defined(HAS_CLWB) || defined(USE_AVX_2) || defined(USE_AVX_512)
/** Use sfence to guarantee memory order on x86. Earlier store operations cannot be reordered beyond this point. */
inline void sfence_barrier() { _mm_sfence(); }
#endif

/** no memory order is guaranteed. */
inline void no_barrier() {}

}  // namespace mema::rw_ops
