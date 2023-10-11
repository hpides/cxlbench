#include "utils.hpp"

#include <numa.h>
#include <sys/mman.h>

#include <filesystem>
#include <fstream>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "json.hpp"
#include "numa.hpp"
#include "test_utils.hpp"

namespace mema::utils {

namespace fs = std::filesystem;

class UtilsTest : public BaseTest {};

/**
 * Verifies whether the first 100'000 zipfian generated values are in between the given boundaries.
 */
TEST_F(UtilsTest, ZipfBound) {
  for (uint32_t i = 0; i < 100'000; i++) {
    const uint64_t value = zipf(0.99, 1000);
    EXPECT_GE(value, 0);
    EXPECT_LT(value, 1000);
  }
}

TEST_F(UtilsTest, CreateResultFileFromConfigFile) {
  const std::filesystem::path config_path = fs::temp_directory_path() / "test.yaml";
  std::ofstream config_file(config_path);

  const fs::path result_dir = fs::temp_directory_path();
  const fs::path result_file = create_result_file(result_dir, config_path);
  ASSERT_TRUE(fs::is_regular_file(result_file));
  EXPECT_THAT(result_file.filename(), testing::StartsWith("test-results-"));
  EXPECT_EQ(result_file.extension().string(), ".json");

  nlohmann::json content;
  std::ifstream results(result_file);
  results >> content;
  EXPECT_TRUE(content.is_array());

  fs::remove(result_file);
  fs::remove(config_path);
}

TEST_F(UtilsTest, CreateResultFileFromConfigDir) {
  const std::filesystem::path config_path = fs::temp_directory_path() / "test-configs";
  fs::create_directories(config_path);

  const fs::path result_dir = fs::temp_directory_path();
  const fs::path result_file = create_result_file(result_dir, config_path);
  ASSERT_TRUE(fs::is_regular_file(result_file));
  EXPECT_THAT(result_file.filename(), testing::StartsWith("test-configs-results-"));
  EXPECT_EQ(result_file.extension().string(), ".json");

  nlohmann::json content;
  std::ifstream results(result_file);
  results >> content;
  EXPECT_TRUE(content.is_array());

  fs::remove(result_file);
  fs::remove(config_path);
}

TEST_F(UtilsTest, AddToResultFile) {
  const std::filesystem::path config_path = fs::temp_directory_path() / "test.yaml";
  std::ofstream config_file(config_path);

  const fs::path result_dir = fs::temp_directory_path();
  const fs::path result_file = create_result_file(result_dir, config_path);
  ASSERT_TRUE(fs::is_regular_file(result_file));

  nlohmann::json result1;
  result1["test"] = true;
  write_benchmark_results(result_file, result1);

  std::ifstream results1(result_file);
  nlohmann::json content1;
  results1 >> content1;
  EXPECT_TRUE(content1.is_array());
  ASSERT_EQ(content1.size(), 1);
  EXPECT_EQ(content1[0].at("test"), true);
  results1.close();

  nlohmann::json result2;
  result2["foo"] = "bar";
  write_benchmark_results(result_file, result2);

  std::ifstream results2(result_file);
  nlohmann::json content2;
  results2 >> content2;
  EXPECT_TRUE(content2.is_array());
  ASSERT_EQ(content2.size(), 2);
  EXPECT_EQ(content2[0].at("test"), true);
  ASSERT_EQ(content2[1].at("foo"), "bar");
  results2.close();

  fs::remove(result_file);
  fs::remove(config_path);
}

TEST_F(UtilsTest, RetrieveCorrectNumaTaskNode) {
  const auto numa_max_node_id = numa_max_node();
  const auto* const cpu_nodes_mask = numa_get_run_node_mask();
  for (auto node_id = NumaNodeID{0}; node_id < numa_max_node_id; ++node_id) {
    if (!numa_bitmask_isbitset(cpu_nodes_mask, node_id)) {
      continue;
    }
    set_task_on_numa_nodes({node_id});
    EXPECT_EQ(get_numa_task_node(), node_id);
  }
}

}  // namespace mema::utils
