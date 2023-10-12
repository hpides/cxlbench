#pragma once

#include <cstddef>

#if defined(HAS_CLWB) || defined(NT_STORES_AVX_2) || defined(NT_STORES_AVX_512)
#include <immintrin.h>
#endif

namespace mema::rw_ops {

// Exactly 64 characters to write in one cache line.
static const char WRITE_DATA[] __attribute__((aligned(64))) =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-";

static constexpr auto CACHE_LINE_SIZE = size_t{64};

using flush_fn = void(char*, const size_t);
using barrier_fn = void();

/** no explicit cache line flush is used. */
inline void no_flush(char* addr, const size_t len) {}

#if defined(HAS_CLWB) || defined(NT_STORES_AVX_2) || defined(NT_STORES_AVX_512)
/** Use sfence to guarantee memory order on x86. Earlier store operations cannot be reordered beyond this point. */
inline void sfence_barrier() { _mm_sfence(); }
#endif

/** no memory order is guaranteed. */
inline void no_barrier() {}

using CharVec64 __attribute__((vector_size(64))) = char;

template <typename Fn, size_t... indices>
void unroll_impl(Fn fn, std::index_sequence<indices...>) {
  // Call fn for all indices
  (void(fn(indices)), ...);
}

template <int LOOP_COUNT, typename Fn>
void unroll(Fn fn) {
  unroll_impl(fn, std::make_index_sequence<LOOP_COUNT>());
}

}  // namespace mema::rw_ops
