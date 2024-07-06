#pragma once

#include "gtest/gtest.h"
#include "json.hpp"

namespace mema {

using BaseTest = ::testing::Test;

#define _PRINT_JSON(json) "Got JSON:\n" << std::setw(2) << json
#define ASSERT_JSON_TRUE(json, assertion) ASSERT_TRUE(json.assertion) << _PRINT_JSON(json)
#define ASSERT_JSON_FALSE(json, assertion) ASSERT_FALSE(json.assertion) << _PRINT_JSON(json)
#define ASSERT_JSON_EQ(json, actual, expected) ASSERT_EQ(json.actual, expected) << _PRINT_JSON(json)

void check_json_result(const nlohmann::json& result_json, uint64_t total_bytes, double expected_bandwidth,
                       uint64_t thread_count, double expected_per_thread_bandwidth, double expected_per_thread_stddev);

std::vector<int> retrieve_page_status(const uint64_t page_count, char* data);

}  // namespace mema
