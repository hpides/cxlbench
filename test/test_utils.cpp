#include "test_utils.hpp"

#include <gmock/gmock-matchers.h>

#include <fstream>

#include "gtest/gtest.h"
#include "read_write_ops.hpp"
#include "utils.hpp"

namespace mema {

void check_json_result(const nlohmann::json& result_json, uint64_t total_bytes, double expected_bandwidth,
                       uint64_t thread_count, double expected_per_thread_bandwidth, double expected_per_thread_stddev) {
  ASSERT_JSON_EQ(result_json, size(), 1);
  ASSERT_JSON_TRUE(result_json, contains("results"));

  const nlohmann::json& results_json = result_json["results"];
  ASSERT_JSON_EQ(results_json, size(), 8);
  ASSERT_JSON_TRUE(results_json, contains("bandwidth"));
  ASSERT_JSON_TRUE(results_json, at("bandwidth").is_number());
  EXPECT_NEAR(results_json.at("bandwidth").get<double>(), expected_bandwidth, 0.001);

  ASSERT_JSON_TRUE(results_json, contains("thread_bandwidth_avg"));
  EXPECT_NEAR(results_json.at("thread_bandwidth_avg").get<double>(), expected_per_thread_bandwidth, 0.001);
  ASSERT_JSON_TRUE(results_json, contains("thread_bandwidth_std_dev"));
  EXPECT_NEAR(results_json.at("thread_bandwidth_std_dev").get<double>(), expected_per_thread_stddev, 0.001);

  ASSERT_JSON_TRUE(results_json, contains("thread_op_latency_avg"));
  ASSERT_JSON_TRUE(results_json, contains("thread_op_latency_std_dev"));

  ASSERT_JSON_TRUE(results_json, contains("execution_time"));
  EXPECT_GT(results_json.at("execution_time").get<double>(), 0);

  ASSERT_JSON_TRUE(results_json, contains("accessed_bytes"));
  EXPECT_EQ(results_json.at("accessed_bytes").get<double>(), total_bytes);

  ASSERT_JSON_TRUE(results_json, contains("threads"));
  EXPECT_EQ(results_json.at("threads").size(), thread_count);
}

}  // namespace mema
