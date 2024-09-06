#ifndef SRC_TYPES_HPP_
#define SRC_TYPES_HPP_

#include <stdint.h>

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <vector>

#define BenchAssert(expr, msg)     \
  if (!static_cast<bool>(expr)) {  \
    std::cerr << msg << std::endl; \
    std::exit(1);                  \
  }                                \
  static_assert(true, "End call of macro with a semicolon")

using u8 = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;
using u64 = uint64_t;
using i8 = int8_t;
using i16 = int16_t;
using i32 = int32_t;
using i64 = int64_t;

using NumaNodeID = u16;
using NumaNodeIDs = std::vector<NumaNodeID>;
using MemoryRegions = std::vector<char*>;
using CoreID = u64;
using CoreIDs = std::vector<CoreID>;
using TimePointMS = std::chrono::time_point<std::chrono::system_clock, std::chrono::milliseconds>;

#endif  // SRC_TYPES_HPP_
