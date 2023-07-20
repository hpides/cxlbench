#include <spdlog/spdlog.h>

#include "gtest/gtest.h"

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  // Disable info logs
  spdlog::set_level(spdlog::level::warn);

  return RUN_ALL_TESTS();
}
