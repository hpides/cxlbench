#include "gtest/gtest.h"
#include "io_operation.hpp"
#include "test_utils.hpp"

namespace mema {

class CustomOperationTest : public BaseTest {};

// Read Operations
TEST_F(CustomOperationTest, ParseCustomRead64) {
  CustomOp op = CustomOp::from_string("r_64");
  EXPECT_EQ(op, (CustomOp{.type = Operation::Read, .size = 64}));
}

TEST_F(CustomOperationTest, ParseCustomRead256) {
  CustomOp op = CustomOp::from_string("r_256");
  EXPECT_EQ(op, (CustomOp{.type = Operation::Read, .size = 256}));
}

TEST_F(CustomOperationTest, ParseCustomRead4096) {
  CustomOp op = CustomOp::from_string("r_4096");
  EXPECT_EQ(op, (CustomOp{.type = Operation::Read, .size = 4096}));
}

TEST_F(CustomOperationTest, ParseBadRead333) { EXPECT_THROW(CustomOp::from_string("r_333"), MemaException); }

TEST_F(CustomOperationTest, ParseBadReadTooShort) { EXPECT_THROW(CustomOp::from_string("r"), MemaException); }

TEST_F(CustomOperationTest, ParseBadReadMissingSize) { EXPECT_THROW(CustomOp::from_string("r_"), MemaException); }

TEST_F(CustomOperationTest, ParseBadReadTooManyArguments) {
  EXPECT_THROW(CustomOp::from_string("r_4096_none"), MemaException);
}

TEST_F(CustomOperationTest, ParseBadReadBadUnderscore) { EXPECT_THROW(CustomOp::from_string("r_p"), MemaException); }

TEST_F(CustomOperationTest, ParseBadReadWhitespace) { EXPECT_THROW(CustomOp::from_string("r p"), MemaException); }

TEST_F(CustomOperationTest, CustomRead64String) {
  EXPECT_EQ((CustomOp{.type = Operation::Read, .size = 64}).to_string(), "r_64");
}

TEST_F(CustomOperationTest, CustomRead128String) {
  EXPECT_EQ((CustomOp{.type = Operation::Read, .size = 128}).to_string(), "r_128");
}

TEST_F(CustomOperationTest, CustomRead256String) {
  EXPECT_EQ((CustomOp{.type = Operation::Read, .size = 256}).to_string(), "r_256");
}

// Write Operations
TEST_F(CustomOperationTest, ParseCustomWrite128None) {
  CustomOp op = CustomOp::from_string("w_128_none");
  EXPECT_EQ(op, (CustomOp{.type = Operation::Write, .size = 128, .flush = FlushInstruction::None}));
}

TEST_F(CustomOperationTest, ParseCustomWrite128NoCache) {
  CustomOp op = CustomOp::from_string("w_128_nocache");
  EXPECT_EQ(op, (CustomOp{.type = Operation::Write, .size = 128, .flush = FlushInstruction::NoCache}));
}

TEST_F(CustomOperationTest, ParseCustomWrite128Cache) {
  CustomOp op = CustomOp::from_string("w_128_cache");
  EXPECT_EQ(op, (CustomOp{.type = Operation::Write, .size = 128, .flush = FlushInstruction::Cache}));
}

TEST_F(CustomOperationTest, ParseCustomWrite256Cache) {
  CustomOp op = CustomOp::from_string("w_256_cache");
  EXPECT_EQ(op, (CustomOp{.type = Operation::Write, .size = 256, .flush = FlushInstruction::Cache}));
}

TEST_F(CustomOperationTest, ParseCustomWrite128Offset) {
  CustomOp op = CustomOp::from_string("w_128_nocache_64");
  EXPECT_EQ(op, (CustomOp{.type = Operation::Write, .size = 128, .flush = FlushInstruction::NoCache, .offset = 64}));
}

TEST_F(CustomOperationTest, ParseCustomWrite128NegativeOffset) {
  CustomOp op = CustomOp::from_string("w_128_cache_-64");
  EXPECT_EQ(op, (CustomOp{.type = Operation::Write, .size = 128, .flush = FlushInstruction::Cache, .offset = -64}));
}

TEST_F(CustomOperationTest, ParseBadWriteOffset) {
  EXPECT_THROW(CustomOp::from_string("w_128_none_333"), MemaException);
}

TEST_F(CustomOperationTest, ParseBadWrite333) { EXPECT_THROW(CustomOp::from_string("w_333_none"), MemaException); }

TEST_F(CustomOperationTest, ParseBadWriteTooShort) { EXPECT_THROW(CustomOp::from_string("w"), MemaException); }

TEST_F(CustomOperationTest, ParseBadWriteMissingSize) { EXPECT_THROW(CustomOp::from_string("w_"), MemaException); }

TEST_F(CustomOperationTest, ParseBadWriteMissingFlushInstruction) {
  EXPECT_THROW(CustomOp::from_string("w_64"), MemaException);
}

TEST_F(CustomOperationTest, ParseBadWriteMissingFlushInstructionWithUnderscore) {
  EXPECT_THROW(CustomOp::from_string("w_64_"), MemaException);
}

TEST_F(CustomOperationTest, ParseBadWriteBadUnderscore) { EXPECT_THROW(CustomOp::from_string("w_p"), MemaException); }

TEST_F(CustomOperationTest, ParseBadWriteWhitespace) { EXPECT_THROW(CustomOp::from_string("w p"), MemaException); }

TEST_F(CustomOperationTest, CustomWrite64NoCacheString) {
  CustomOp op{.type = Operation::Write, .size = 64, .flush = FlushInstruction::NoCache};
  EXPECT_EQ(op.to_string(), "w_64_nocache");
}

TEST_F(CustomOperationTest, CustomWrite128CacheString) {
  CustomOp op{.type = Operation::Write, .size = 128, .flush = FlushInstruction::Cache};
  EXPECT_EQ(op.to_string(), "w_128_cache");
}

TEST_F(CustomOperationTest, CustomWrite256CacheString) {
  CustomOp op{.type = Operation::Write, .size = 256, .flush = FlushInstruction::Cache};
  EXPECT_EQ(op.to_string(), "w_256_cache");
}

TEST_F(CustomOperationTest, CustomWrite4096NoneString) {
  CustomOp op{.type = Operation::Write, .size = 4096, .flush = FlushInstruction::None};
  EXPECT_EQ(op.to_string(), "w_4096_none");
}

TEST_F(CustomOperationTest, CustomWrite128OffsetString) {
  CustomOp op{.type = Operation::Write, .size = 128, .flush = FlushInstruction::Cache, .offset = 128};
  EXPECT_EQ(op.to_string(), "w_128_cache_128");
}

TEST_F(CustomOperationTest, CustomWrite128NegativeOffsetString) {
  CustomOp op{.type = Operation::Write, .size = 128, .flush = FlushInstruction::Cache, .offset = -64};
  EXPECT_EQ(op.to_string(), "w_128_cache_-64");
}

TEST_F(CustomOperationTest, BadChainStartsWithWrite) {
  std::vector<CustomOp> ops = {CustomOp{.type = Operation::Write}, CustomOp{.type = Operation::Read}};
  EXPECT_FALSE(CustomOp::validate(ops));
}

}  // namespace mema
