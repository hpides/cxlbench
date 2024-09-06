#include "access_batch.hpp"
#include "gtest/gtest.h"
#include "test_utils.hpp"

namespace mema {

class CustomOperationTest : public BaseTest {};

// Read Operations
TEST_F(CustomOperationTest, ParseCustomRead64) {
  CustomOp op = CustomOp::from_string("r_64");
  EXPECT_EQ(op, (CustomOp{.memory_type = MemoryType::Primary, .type = Operation::Read, .size = 64}));
}

TEST_F(CustomOperationTest, ParseCustomMemoryRegionPrefix) {
  CustomOp op = CustomOp::from_string("m0_r_64");
  EXPECT_EQ(op, (CustomOp{.memory_type = MemoryType::Primary, .type = Operation::Read, .size = 64}));
  op = CustomOp::from_string("m1_r_64");
  EXPECT_EQ(op, (CustomOp{.memory_type = MemoryType::Secondary, .type = Operation::Read, .size = 64}));
  op = CustomOp::from_string("m2_r_64");
  EXPECT_EQ(op, (CustomOp{.memory_type = MemoryType::Secondary, .type = Operation::Read, .size = 64}));
  op = CustomOp::from_string("m42_r_64");
  EXPECT_EQ(op, (CustomOp{.memory_type = MemoryType::Secondary, .type = Operation::Read, .size = 64}));
}

TEST_F(CustomOperationTest, ParseCustomRead256) {
  CustomOp op = CustomOp::from_string("r_256");
  EXPECT_EQ(op, (CustomOp{.type = Operation::Read, .size = 256}));
  op = CustomOp::from_string("m0_r_256");
  EXPECT_EQ(op, (CustomOp{.type = Operation::Read, .size = 256}));
}

TEST_F(CustomOperationTest, ParseCustomRead4096) {
  CustomOp op = CustomOp::from_string("r_4096");
  EXPECT_EQ(op, (CustomOp{.type = Operation::Read, .size = 4096}));
  op = CustomOp::from_string("m0_r_4096");
  EXPECT_EQ(op, (CustomOp{.type = Operation::Read, .size = 4096}));
}

TEST_F(CustomOperationTest, ParseBadRead333) {
  EXPECT_THROW(CustomOp::from_string("r_333"), MemaException);
  EXPECT_THROW(CustomOp::from_string("m0_r_333"), MemaException);
}

TEST_F(CustomOperationTest, ParseBadReadTooShort) {
  EXPECT_THROW(CustomOp::from_string("r"), MemaException);
  EXPECT_THROW(CustomOp::from_string("m0_r"), MemaException);
}

TEST_F(CustomOperationTest, ParseBadReadMissingSize) {
  EXPECT_THROW(CustomOp::from_string("r_"), MemaException);
  EXPECT_THROW(CustomOp::from_string("m0_r_"), MemaException);
}

TEST_F(CustomOperationTest, ParseBadReadTooManyArguments) {
  EXPECT_THROW(CustomOp::from_string("r_4096_none"), MemaException);
  EXPECT_THROW(CustomOp::from_string("m0_r_4096_none"), MemaException);
}

TEST_F(CustomOperationTest, ParseBadReadBadUnderscore) { EXPECT_THROW(CustomOp::from_string("r_p"), MemaException); }

TEST_F(CustomOperationTest, ParseBadReadWhitespace) { EXPECT_THROW(CustomOp::from_string("r p"), MemaException); }

TEST_F(CustomOperationTest, CustomRead64String) {
  EXPECT_EQ((CustomOp{.type = Operation::Read, .size = 64}).to_string(), "m0_r_64");
}

TEST_F(CustomOperationTest, CustomRead128String) {
  EXPECT_EQ((CustomOp{.type = Operation::Read, .size = 128}).to_string(), "m0_r_128");
}

TEST_F(CustomOperationTest, CustomRead256String) {
  EXPECT_EQ((CustomOp{.type = Operation::Read, .size = 256}).to_string(), "m0_r_256");
}

TEST_F(CustomOperationTest, CustomReadSecondaryRegionString) {
  EXPECT_EQ((CustomOp{.memory_type = MemoryType::Secondary, .type = Operation::Read, .size = 256}).to_string(),
            "m1_r_256");
}

// Write Operations
TEST_F(CustomOperationTest, ParseCustomWrite128None) {
  CustomOp op = CustomOp::from_string("w_128_none");
  EXPECT_EQ(op, (CustomOp{.type = Operation::Write, .size = 128, .cache_fn = CacheInstruction::None}));
  op = CustomOp::from_string("m1_w_128_none");
  EXPECT_EQ(op, (CustomOp{.memory_type = MemoryType::Secondary,
                          .type = Operation::Write,
                          .size = 128,
                          .cache_fn = CacheInstruction::None}));
}

TEST_F(CustomOperationTest, ParseCustomWrite128NoCache) {
  CustomOp op = CustomOp::from_string("w_128_nocache");
  EXPECT_EQ(op, (CustomOp{.type = Operation::Write, .size = 128, .cache_fn = CacheInstruction::NoCache}));
  op = CustomOp::from_string("m1_w_128_nocache");
  EXPECT_EQ(op, (CustomOp{.memory_type = MemoryType::Secondary,
                          .type = Operation::Write,
                          .size = 128,
                          .cache_fn = CacheInstruction::NoCache}));
}

TEST_F(CustomOperationTest, ParseCustomWrite128Cache) {
  CustomOp op = CustomOp::from_string("w_128_cache");
  EXPECT_EQ(op, (CustomOp{.type = Operation::Write, .size = 128, .cache_fn = CacheInstruction::Cache}));
  op = CustomOp::from_string("m1_w_128_cache");
  EXPECT_EQ(op, (CustomOp{.memory_type = MemoryType::Secondary,
                          .type = Operation::Write,
                          .size = 128,
                          .cache_fn = CacheInstruction::Cache}));
}

TEST_F(CustomOperationTest, ParseCustomWrite256Cache) {
  CustomOp op = CustomOp::from_string("w_256_cache");
  EXPECT_EQ(op, (CustomOp{.type = Operation::Write, .size = 256, .cache_fn = CacheInstruction::Cache}));
  op = CustomOp::from_string("w_256_cache");
  EXPECT_EQ(op, (CustomOp{.memory_type = MemoryType::Secondary,
                          .type = Operation::Write,
                          .size = 256,
                          .cache_fn = CacheInstruction::Cache}));
}

TEST_F(CustomOperationTest, ParseCustomWrite128Offset) {
  CustomOp op = CustomOp::from_string("w_128_nocache_64");
  EXPECT_EQ(op, (CustomOp{.type = Operation::Write, .size = 128, .cache_fn = CacheInstruction::NoCache, .offset = 64}));
  op = CustomOp::from_string("m1_w_128_nocache_64");
  EXPECT_EQ(op, (CustomOp{.memory_type = MemoryType::Secondary,
                          .type = Operation::Write,
                          .size = 128,
                          .cache_fn = CacheInstruction::NoCache,
                          .offset = 64}));
}

TEST_F(CustomOperationTest, ParseCustomWrite128NegativeOffset) {
  CustomOp op = CustomOp::from_string("w_128_cache_-64");
  EXPECT_EQ(op, (CustomOp{.type = Operation::Write, .size = 128, .cache_fn = CacheInstruction::Cache, .offset = -64}));
  op = CustomOp::from_string("m1_w_128_cache_-64");
  EXPECT_EQ(op, (CustomOp{.memory_type = MemoryType::Secondary,
                          .type = Operation::Write,
                          .size = 128,
                          .cache_fn = CacheInstruction::Cache,
                          .offset = -64}));
}

TEST_F(CustomOperationTest, ParseBadWriteOffset) {
  EXPECT_THROW(CustomOp::from_string("w_128_none_333"), MemaException);
  EXPECT_THROW(CustomOp::from_string("m0_w_128_none_333"), MemaException);
  EXPECT_THROW(CustomOp::from_string("m1_w_128_none_333"), MemaException);
}

TEST_F(CustomOperationTest, ParseBadWrite333) {
  EXPECT_THROW(CustomOp::from_string("w_333_none"), MemaException);
  EXPECT_THROW(CustomOp::from_string("m0_w_333_none"), MemaException);
  EXPECT_THROW(CustomOp::from_string("m1_w_333_none"), MemaException);
}

TEST_F(CustomOperationTest, ParseBadWriteTooShort) {
  EXPECT_THROW(CustomOp::from_string("w"), MemaException);
  EXPECT_THROW(CustomOp::from_string("m0_w"), MemaException);
}

TEST_F(CustomOperationTest, ParseBadWriteMissingSize) {
  EXPECT_THROW(CustomOp::from_string("w_"), MemaException);
  EXPECT_THROW(CustomOp::from_string("m0_w_"), MemaException);
}

TEST_F(CustomOperationTest, ParseBadWriteMissingCacheInstruction) {
  EXPECT_THROW(CustomOp::from_string("w_64"), MemaException);
  EXPECT_THROW(CustomOp::from_string("m0_w_64"), MemaException);
}

TEST_F(CustomOperationTest, ParseBadWriteMissingCacheInstructionWithUnderscore) {
  EXPECT_THROW(CustomOp::from_string("w_64_"), MemaException);
  EXPECT_THROW(CustomOp::from_string("m0_w_64_"), MemaException);
}

TEST_F(CustomOperationTest, ParseBadWriteBadUnderscore) {
  EXPECT_THROW(CustomOp::from_string("w_p"), MemaException);
  EXPECT_THROW(CustomOp::from_string("m0_w_p"), MemaException);
}

TEST_F(CustomOperationTest, ParseBadWriteWhitespace) {
  EXPECT_THROW(CustomOp::from_string("w p"), MemaException);
  EXPECT_THROW(CustomOp::from_string("m0_w p"), MemaException);
}

TEST_F(CustomOperationTest, CustomWrite64NoCacheString) {
  CustomOp op{.type = Operation::Write, .size = 64, .cache_fn = CacheInstruction::NoCache};
  EXPECT_EQ(op.to_string(), "m0_w_64_nocache");
  op = CustomOp{.memory_type = MemoryType::Secondary,
                .type = Operation::Write,
                .size = 64,
                .cache_fn = CacheInstruction::NoCache};
  EXPECT_EQ(op.to_string(), "m1_w_64_nocache");
}

TEST_F(CustomOperationTest, CustomWrite128CacheString) {
  CustomOp op{.type = Operation::Write, .size = 128, .cache_fn = CacheInstruction::Cache};
  EXPECT_EQ(op.to_string(), "m0_w_128_cache");
  op = CustomOp{
      .memory_type = MemoryType::Secondary, .type = Operation::Write, .size = 128, .cache_fn = CacheInstruction::Cache};
  EXPECT_EQ(op.to_string(), "m1_w_128_cache");
}

TEST_F(CustomOperationTest, CustomWrite256CacheString) {
  CustomOp op{.type = Operation::Write, .size = 256, .cache_fn = CacheInstruction::Cache};
  EXPECT_EQ(op.to_string(), "m0_w_256_cache");
  op = CustomOp{
      .memory_type = MemoryType::Secondary, .type = Operation::Write, .size = 256, .cache_fn = CacheInstruction::Cache};
  EXPECT_EQ(op.to_string(), "m1_w_256_cache");
}

TEST_F(CustomOperationTest, CustomWrite4096NoneString) {
  CustomOp op{.type = Operation::Write, .size = 4096, .cache_fn = CacheInstruction::None};
  EXPECT_EQ(op.to_string(), "m0_w_4096_none");
  op = CustomOp{
      .memory_type = MemoryType::Secondary, .type = Operation::Write, .size = 4096, .cache_fn = CacheInstruction::None};
  EXPECT_EQ(op.to_string(), "m1_w_4096_none");
}

TEST_F(CustomOperationTest, CustomWrite128OffsetString) {
  CustomOp op{.type = Operation::Write, .size = 128, .cache_fn = CacheInstruction::Cache, .offset = 128};
  EXPECT_EQ(op.to_string(), "m0_w_128_cache_128");
  op = CustomOp{.memory_type = MemoryType::Secondary,
                .type = Operation::Write,
                .size = 128,
                .cache_fn = CacheInstruction::Cache,
                .offset = 128};
  EXPECT_EQ(op.to_string(), "m1_w_128_cache_128");
}

TEST_F(CustomOperationTest, CustomWrite128NegativeOffsetString) {
  CustomOp op{.type = Operation::Write, .size = 128, .cache_fn = CacheInstruction::Cache, .offset = -64};
  EXPECT_EQ(op.to_string(), "m0_w_128_cache_-64");
  op = CustomOp{.memory_type = MemoryType::Secondary,
                .type = Operation::Write,
                .size = 128,
                .cache_fn = CacheInstruction::Cache,
                .offset = -64};
  EXPECT_EQ(op.to_string(), "m1_w_128_cache_-64");
}

TEST_F(CustomOperationTest, BadChainStartsWithWrite) {
  std::vector<CustomOp> ops = {CustomOp{.type = Operation::Write}, CustomOp{.type = Operation::Read}};
  EXPECT_THROW(CustomOp::validate(ops), MemaException);
}

TEST_F(CustomOperationTest, BadSecondaryChainStartsWithWrite) {
  std::vector<CustomOp> ops = {CustomOp{.type = Operation::Read}, CustomOp{.type = Operation::Write},
                               CustomOp{.memory_type = MemoryType::Secondary, .type = Operation::Write},
                               CustomOp{.memory_type = MemoryType::Secondary, .type = Operation::Read}};
  EXPECT_THROW(CustomOp::validate(ops), MemaException);
}

TEST_F(CustomOperationTest, BadPrimaryChainStartsWithWriteAfterSecondary) {
  std::vector<CustomOp> ops = {CustomOp{.memory_type = MemoryType::Secondary, .type = Operation::Read},
                               CustomOp{.memory_type = MemoryType::Secondary, .type = Operation::Write},
                               CustomOp{.type = Operation::Write}, CustomOp{.type = Operation::Read}};
  EXPECT_THROW(CustomOp::validate(ops), MemaException);
}

TEST_F(CustomOperationTest, PrimarySecondaryMemoryAccessChains) {
  std::vector<CustomOp> ops = {CustomOp{.type = Operation::Read},
                               CustomOp{.type = Operation::Write},
                               CustomOp{.memory_type = MemoryType::Secondary, .type = Operation::Read},
                               CustomOp{.memory_type = MemoryType::Secondary, .type = Operation::Write},
                               CustomOp{.memory_type = MemoryType::Secondary, .type = Operation::Write},
                               CustomOp{.type = Operation::Read},
                               CustomOp{.type = Operation::Write}};
  EXPECT_NO_THROW(CustomOp::validate(ops));
}

}  // namespace mema
