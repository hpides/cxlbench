#pragma once

#include <cstddef>

#if defined(HAS_CLWB) || defined(USE_AVX_2) || defined(USE_AVX_512)
#include <immintrin.h>
#endif

namespace mema::rw_ops {

// Exactly 64 characters to write in one cache line.
static const char WRITE_DATA[] __attribute__((aligned(64))) =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-";

static constexpr auto CACHE_LINE_SIZE = size_t{64};

// We use platform-independent vector intrinsics (compiler intrinsics) to avoid writing platform-specific SIMD
// intrinsics (only for char data type; see definition of CharVec). VECTOR_SIZE defines the vector size and, in the case
// of type char, the number of Bytes being accessed. 64 B (cache line size) is the base access size in this benchmark
// tool. We need a certain number of vectorized memory accesses to access 64 B. This number is determined by
// CACHE_LINE_FACTOR.

#ifdef USE_AVX_2
static constexpr size_t VECTOR_SIZE = 32;
#else
static constexpr size_t VECTOR_SIZE = 64;
#endif

static constexpr size_t CACHE_LINE_FACTOR = CACHE_LINE_SIZE / VECTOR_SIZE;

using CharVec __attribute__((vector_size(VECTOR_SIZE))) = char;

using flush_fn = void(char*, const size_t);
using barrier_fn = void();

/** no explicit cache line flush is used. */
inline void no_flush(char* addr, const size_t len) {}

#if defined(HAS_CLWB) || defined(USE_AVX_2) || defined(USE_AVX_512)
/** Use sfence to guarantee memory order on x86. Earlier store operations cannot be reordered beyond this point. */
inline void sfence_barrier() { _mm_sfence(); }
#endif

/** no memory order is guaranteed. */
inline void no_barrier() {}

template <typename Fn, size_t... indices>
void unroll_impl(Fn fn, std::index_sequence<indices...>) {
  // Call fn for sequence of passed indices.
  (fn(indices), ...);
}

template <int LOOP_COUNT, typename Fn>
void unroll(Fn fn) {
  // Call unroll_impl with passed function and a compile-time sequence of integers [0, ..., LOOP_COUNT].
  unroll_impl(fn, std::make_index_sequence<LOOP_COUNT>());
}

}  // namespace mema::rw_ops
