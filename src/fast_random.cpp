#include "fast_random.hpp"

#include <cstdint>

namespace mema {

thread_local __uint128_t g_lehmer64_state = 0;

}  // namespace mema
