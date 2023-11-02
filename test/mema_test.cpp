#include <spdlog/spdlog.h>

#include "gtest/gtest.h"

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  // Disable info logs
  spdlog::set_level(spdlog::level::critical);
  // Use the following line to debug failing test cases.
  // spdlog::set_level(spdlog::level::debug);

  return RUN_ALL_TESTS();
}
