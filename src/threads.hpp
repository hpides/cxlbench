#pragma once

#include "benchmark.hpp"

namespace mema {

// Returns core ids on which the current thread is allowed to run on.
CoreIDs allowed_thread_core_ids();

void pin_thread_to_cores(const CoreIDs& core_ids);

}  // namespace mema
