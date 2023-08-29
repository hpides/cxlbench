#pragma once

#include <cstddef>

namespace mema::rw_ops {

// Exactly 64 characters to write in one cache line.
static const char WRITE_DATA[] __attribute__((aligned(64))) =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-";

static constexpr auto CACHE_LINE_SIZE = size_t{64};

using flush_fn = void(char*, const size_t);
using barrier_fn = void();

/** no explicit cache line flush is used. */
inline void no_flush(char* addr, const size_t len) {}

/** Use sfence to guarantee memory order on x86. Earlier store operations cannot be reordered beyond this point. */
inline void sfence_barrier() { _mm_sfence(); }

/** no memory order is guaranteed. */
inline void no_barrier() {}

}  // namespace mema::rw_ops
