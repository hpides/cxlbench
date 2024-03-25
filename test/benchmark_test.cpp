#include "benchmark.hpp"

#include <fcntl.h>
#include <numa.h>

#include <fstream>

#include "gmock/gmock-matchers.h"
#include "gtest/gtest.h"
#include "numa.hpp"
#include "parallel_benchmark.hpp"
#include "single_benchmark.hpp"
#include "test_utils.hpp"

namespace mema {

using ::testing::ElementsAre;

constexpr size_t TEST_DATA_SIZE = 1 * MiB;              // 1 MiB
constexpr size_t TEST_CHUNK_SIZE = TEST_DATA_SIZE / 8;  // 128 KiB

class BenchmarkTest : public BaseTest {
 protected:
  void SetUp() override {
    base_config_.memory_region_size = TEST_DATA_SIZE;
    base_config_.min_io_chunk_size = TEST_CHUNK_SIZE;
    base_config_.numa_memory_nodes = {0};
    base_config_.numa_task_nodes = {0};
  }

  BenchmarkConfig base_config_{};
  std::vector<std::unique_ptr<BenchmarkExecution>> base_executions_{};
  std::vector<std::unique_ptr<BenchmarkResult>> base_results_{};
  const std::string bm_name_ = "test_bm";
};

TEST_F(BenchmarkTest, CreateSingleBenchmark) {
  ASSERT_NO_THROW(SingleBenchmark("test_bm1", base_config_, std::move(base_executions_), std::move(base_results_)));
}

TEST_F(BenchmarkTest, CreateParallelBenchmark) {
  ASSERT_NO_THROW(ParallelBenchmark("test_bm1", "sub_bm_1_1", "sub_bm_1_2", base_config_, base_config_,
                                    std::move(base_executions_), std::move(base_results_)));
}

TEST_F(BenchmarkTest, SetUpSingleThread) {
  base_config_.number_threads = 1;
  base_config_.access_size = 256;

  base_executions_.reserve(1);
  base_executions_.push_back(std::make_unique<BenchmarkExecution>());
  base_results_.reserve(1);
  base_results_.push_back(std::make_unique<BenchmarkResult>(base_config_));
  SingleBenchmark bm{bm_name_, base_config_, std::move(base_executions_), std::move(base_results_)};
  bm.generate_data();
  bm.set_up();

  const std::vector<ThreadRunConfig>& thread_configs = bm.get_thread_configs()[0];
  ASSERT_EQ(thread_configs.size(), 1);
  const ThreadRunConfig& thread_config = thread_configs[0];

  EXPECT_EQ(thread_config.thread_idx, 0);
  EXPECT_EQ(thread_config.thread_count_per_partition, 1);
  EXPECT_EQ(thread_config.partition_size, TEST_DATA_SIZE);
  EXPECT_EQ(thread_config.ops_count_per_chunk, TEST_CHUNK_SIZE / 256);
  EXPECT_EQ(thread_config.chunk_count, 8);
  EXPECT_EQ(thread_config.partition_start_addr, bm.get_data()[0]);
  EXPECT_EQ(&thread_config.config, &bm.get_benchmark_configs()[0]);

  const std::vector<ExecutionDuration>& op_durations = bm.get_benchmark_results()[0]->total_operation_durations;
  ASSERT_EQ(op_durations.size(), 1);
  EXPECT_EQ(thread_config.total_operation_duration, &op_durations[0]);

  const std::vector<uint64_t>& op_sizes = bm.get_benchmark_results()[0]->total_operation_sizes;
  ASSERT_EQ(op_sizes.size(), 1);
  EXPECT_EQ(thread_config.total_operation_size, &op_sizes[0]);

  bm.get_benchmark_results()[0]->config.validate();
}

TEST_F(BenchmarkTest, SetUpMultiThreadCustomPartition) {
  const size_t thread_count = 4;
  base_config_.number_threads = thread_count;
  base_config_.number_partitions = 2;
  base_config_.access_size = 512;

  base_executions_.reserve(1);
  base_executions_.push_back(std::make_unique<BenchmarkExecution>());
  base_results_.reserve(1);
  base_results_.push_back(std::make_unique<BenchmarkResult>(base_config_));
  SingleBenchmark bm{bm_name_, base_config_, std::move(base_executions_), std::move(base_results_)};
  bm.generate_data();
  bm.set_up();

  const size_t partition_size = TEST_DATA_SIZE / 2;
  const std::vector<ThreadRunConfig>& thread_configs = bm.get_thread_configs()[0];
  ASSERT_EQ(thread_configs.size(), thread_count);
  const ThreadRunConfig& thread_config0 = thread_configs[0];
  const ThreadRunConfig& thread_config1 = thread_configs[1];
  const ThreadRunConfig& thread_config2 = thread_configs[2];
  const ThreadRunConfig& thread_config3 = thread_configs[3];

  EXPECT_EQ(thread_config0.thread_idx, 0);
  EXPECT_EQ(thread_config1.thread_idx, 1);
  EXPECT_EQ(thread_config2.thread_idx, 2);
  EXPECT_EQ(thread_config3.thread_idx, 3);

  EXPECT_EQ(thread_config0.partition_start_addr, bm.get_data()[0]);
  EXPECT_EQ(thread_config1.partition_start_addr, bm.get_data()[0]);
  EXPECT_EQ(thread_config2.partition_start_addr, bm.get_data()[0] + partition_size);
  EXPECT_EQ(thread_config3.partition_start_addr, bm.get_data()[0] + partition_size);

  const std::vector<ExecutionDuration>& op_durations = bm.get_benchmark_results()[0]->total_operation_durations;
  ASSERT_EQ(op_durations.size(), thread_count);
  EXPECT_EQ(thread_config0.total_operation_duration, &op_durations[0]);
  EXPECT_EQ(thread_config1.total_operation_duration, &op_durations[1]);
  EXPECT_EQ(thread_config2.total_operation_duration, &op_durations[2]);
  EXPECT_EQ(thread_config3.total_operation_duration, &op_durations[3]);

  const std::vector<uint64_t>& op_sizes = bm.get_benchmark_results()[0]->total_operation_sizes;
  ASSERT_EQ(op_sizes.size(), thread_count);
  EXPECT_EQ(thread_config0.total_operation_size, &op_sizes[0]);
  EXPECT_EQ(thread_config1.total_operation_size, &op_sizes[1]);
  EXPECT_EQ(thread_config2.total_operation_size, &op_sizes[2]);
  EXPECT_EQ(thread_config3.total_operation_size, &op_sizes[3]);

  // These values are the same for all threads
  for (const ThreadRunConfig& tc : thread_configs) {
    EXPECT_EQ(tc.thread_count_per_partition, 2);
    EXPECT_EQ(tc.partition_size, partition_size);
    EXPECT_EQ(tc.ops_count_per_chunk, TEST_CHUNK_SIZE / 512);
    EXPECT_EQ(tc.chunk_count, 8);
    EXPECT_EQ(&tc.config, &bm.get_benchmark_configs()[0]);
  }
  bm.get_benchmark_results()[0]->config.validate();
}

TEST_F(BenchmarkTest, SetUpMultiThreadDefaultPartition) {
  const size_t thread_count = 4;
  base_config_.number_threads = thread_count;
  base_config_.number_partitions = 0;
  base_config_.access_size = 256;

  base_executions_.reserve(1);
  base_executions_.push_back(std::make_unique<BenchmarkExecution>());
  base_results_.reserve(1);
  base_results_.push_back(std::make_unique<BenchmarkResult>(base_config_));
  SingleBenchmark bm{bm_name_, base_config_, std::move(base_executions_), std::move(base_results_)};
  bm.generate_data();
  bm.set_up();

  const size_t partition_size = TEST_DATA_SIZE / 4;
  const std::vector<ThreadRunConfig>& thread_configs = bm.get_thread_configs()[0];
  ASSERT_EQ(thread_configs.size(), thread_count);
  const ThreadRunConfig& thread_config0 = thread_configs[0];
  const ThreadRunConfig& thread_config1 = thread_configs[1];
  const ThreadRunConfig& thread_config2 = thread_configs[2];
  const ThreadRunConfig& thread_config3 = thread_configs[3];

  EXPECT_EQ(thread_config0.thread_idx, 0);
  EXPECT_EQ(thread_config1.thread_idx, 1);
  EXPECT_EQ(thread_config2.thread_idx, 2);
  EXPECT_EQ(thread_config3.thread_idx, 3);

  EXPECT_EQ(thread_config0.partition_start_addr, bm.get_data()[0] + (0 * partition_size));
  EXPECT_EQ(thread_config1.partition_start_addr, bm.get_data()[0] + (1 * partition_size));
  EXPECT_EQ(thread_config2.partition_start_addr, bm.get_data()[0] + (2 * partition_size));
  EXPECT_EQ(thread_config3.partition_start_addr, bm.get_data()[0] + (3 * partition_size));

  const std::vector<ExecutionDuration>& op_durations = bm.get_benchmark_results()[0]->total_operation_durations;
  ASSERT_EQ(op_durations.size(), thread_count);
  EXPECT_EQ(thread_config0.total_operation_duration, &op_durations[0]);
  EXPECT_EQ(thread_config1.total_operation_duration, &op_durations[1]);
  EXPECT_EQ(thread_config2.total_operation_duration, &op_durations[2]);
  EXPECT_EQ(thread_config3.total_operation_duration, &op_durations[3]);

  const std::vector<uint64_t>& op_sizes = bm.get_benchmark_results()[0]->total_operation_sizes;
  ASSERT_EQ(op_sizes.size(), thread_count);
  EXPECT_EQ(thread_config0.total_operation_size, &op_sizes[0]);
  EXPECT_EQ(thread_config1.total_operation_size, &op_sizes[1]);
  EXPECT_EQ(thread_config2.total_operation_size, &op_sizes[2]);
  EXPECT_EQ(thread_config3.total_operation_size, &op_sizes[3]);

  // These values are the same for all threads
  for (const ThreadRunConfig& tc : thread_configs) {
    EXPECT_EQ(tc.thread_count_per_partition, 1);
    EXPECT_EQ(tc.partition_size, partition_size);
    EXPECT_EQ(tc.ops_count_per_chunk, TEST_CHUNK_SIZE / 256);
    EXPECT_EQ(tc.chunk_count, 8);
    EXPECT_EQ(&tc.config, &bm.get_benchmark_configs()[0]);
  }
  bm.get_benchmark_results()[0]->config.validate();
}

TEST_F(BenchmarkTest, RunSingeThreadRead) {
  const size_t num_ops = TEST_DATA_SIZE / 256;
  base_config_.number_threads = 1;
  base_config_.access_size = 256;
  base_config_.operation = Operation::Read;
  base_config_.memory_region_size = 256 * num_ops;

  base_executions_.reserve(1);
  base_executions_.push_back(std::make_unique<BenchmarkExecution>());
  base_results_.reserve(1);
  base_results_.push_back(std::make_unique<BenchmarkResult>(base_config_));
  SingleBenchmark bm{bm_name_, base_config_, std::move(base_executions_), std::move(base_results_)};

  const auto start_test_ts = std::chrono::steady_clock::now();

  bm.generate_data();
  bm.set_up();
  bm.run();

  const BenchmarkResult& result = *bm.get_benchmark_results()[0];

  const std::vector<ExecutionDuration>& op_durations = bm.get_benchmark_results()[0]->total_operation_durations;
  ASSERT_EQ(op_durations.size(), 1);
  EXPECT_GT(op_durations[0].begin, start_test_ts);
  EXPECT_GT(op_durations[0].end, start_test_ts);
  EXPECT_LT(op_durations[0].begin, op_durations[0].end);

  const std::vector<uint64_t>& op_sizes = bm.get_benchmark_results()[0]->total_operation_sizes;
  EXPECT_THAT(op_sizes, ElementsAre(TEST_DATA_SIZE));
}

TEST_F(BenchmarkTest, RunSingleThreadWrite) {
  const size_t num_ops = TEST_DATA_SIZE / 64;
  const size_t total_size = 64 * num_ops;
  base_config_.number_threads = 1;
  base_config_.access_size = 64;
  base_config_.operation = Operation::Write;
  base_config_.memory_region_size = total_size;

  base_executions_.reserve(1);
  base_executions_.push_back(std::make_unique<BenchmarkExecution>());
  base_results_.reserve(1);
  base_results_.push_back(std::make_unique<BenchmarkResult>(base_config_));
  SingleBenchmark bm{bm_name_, base_config_, std::move(base_executions_), std::move(base_results_)};

  const auto start_test_ts = std::chrono::steady_clock::now();

  bm.generate_data();
  bm.set_up();
  bm.run();

  const BenchmarkResult& result = *bm.get_benchmark_results()[0];

  const std::vector<ExecutionDuration>& op_durations = bm.get_benchmark_results()[0]->total_operation_durations;
  ASSERT_EQ(op_durations.size(), 1);
  EXPECT_GT(op_durations[0].begin, start_test_ts);
  EXPECT_GT(op_durations[0].end, start_test_ts);
  EXPECT_LT(op_durations[0].begin, op_durations[0].end);

  const std::vector<uint64_t>& op_sizes = bm.get_benchmark_results()[0]->total_operation_sizes;
  EXPECT_THAT(op_sizes, ElementsAre(total_size));
}

TEST_F(BenchmarkTest, RunMultiThreadRead) {
  const size_t num_ops = TEST_DATA_SIZE / 1024;
  const size_t thread_count = 4;
  base_config_.number_threads = thread_count;
  base_config_.access_size = 1024;
  base_config_.operation = Operation::Read;
  base_config_.memory_region_size = 1024 * num_ops;

  base_executions_.reserve(1);
  base_executions_.push_back(std::make_unique<BenchmarkExecution>());
  base_results_.reserve(1);
  base_results_.push_back(std::make_unique<BenchmarkResult>(base_config_));
  SingleBenchmark bm{bm_name_, base_config_, std::move(base_executions_), std::move(base_results_)};

  const auto start_test_ts = std::chrono::steady_clock::now();

  bm.generate_data();
  bm.set_up();
  bm.run();

  const BenchmarkResult& result = *bm.get_benchmark_results()[0];

  const std::vector<ExecutionDuration>& op_durations = bm.get_benchmark_results()[0]->total_operation_durations;
  EXPECT_EQ(op_durations.size(), thread_count);
  for (ExecutionDuration duration : op_durations) {
    EXPECT_GT(duration.begin, start_test_ts);
    EXPECT_GT(duration.end, start_test_ts);
    EXPECT_LE(duration.begin, duration.end);
  }

  const std::vector<uint64_t>& op_sizes = bm.get_benchmark_results()[0]->total_operation_sizes;
  EXPECT_EQ(op_durations.size(), thread_count);
  EXPECT_EQ(std::accumulate(op_sizes.begin(), op_sizes.end(), 0ul), TEST_DATA_SIZE);
  for (uint64_t size : op_sizes) {
    EXPECT_EQ(size % TEST_CHUNK_SIZE, 0);
  }
}

TEST_F(BenchmarkTest, RunMultiThreadWrite) {
  const size_t num_ops = TEST_DATA_SIZE / 512;
  const size_t thread_count = 16;
  const size_t total_size = 512 * num_ops;
  base_config_.number_threads = thread_count;
  base_config_.access_size = 512;
  base_config_.operation = Operation::Read;
  base_config_.memory_region_size = total_size;

  base_executions_.reserve(1);
  base_executions_.push_back(std::make_unique<BenchmarkExecution>());
  base_results_.reserve(1);
  base_results_.push_back(std::make_unique<BenchmarkResult>(base_config_));
  SingleBenchmark bm{bm_name_, base_config_, std::move(base_executions_), std::move(base_results_)};

  const auto start_test_ts = std::chrono::steady_clock::now();

  bm.generate_data();
  bm.set_up();
  bm.run();

  const BenchmarkResult& result = *bm.get_benchmark_results()[0];

  const std::vector<ExecutionDuration>& op_durations = bm.get_benchmark_results()[0]->total_operation_durations;
  EXPECT_EQ(op_durations.size(), thread_count);
  for (ExecutionDuration duration : op_durations) {
    EXPECT_GT(duration.begin, start_test_ts);
    EXPECT_GT(duration.end, start_test_ts);
    EXPECT_LE(duration.begin, duration.end);
  }

  const std::vector<uint64_t>& op_sizes = bm.get_benchmark_results()[0]->total_operation_sizes;
  EXPECT_EQ(op_sizes.size(), 16);
  EXPECT_EQ(std::accumulate(op_sizes.begin(), op_sizes.end(), 0ul), TEST_DATA_SIZE);
  for (uint64_t size : op_sizes) {
    EXPECT_EQ(size % TEST_CHUNK_SIZE, 0);
  }
}

TEST_F(BenchmarkTest, RunMultiThreadReadDesc) {
  const size_t num_ops = TEST_DATA_SIZE / 1024;
  const size_t thread_count = 4;
  base_config_.number_threads = thread_count;
  base_config_.access_size = 1024;
  base_config_.operation = Operation::Read;
  base_config_.memory_region_size = 1024 * num_ops;

  base_config_.exec_mode = Mode::Sequential_Desc;
  base_executions_.reserve(1);
  base_executions_.push_back(std::make_unique<BenchmarkExecution>());
  base_results_.reserve(1);
  base_results_.push_back(std::make_unique<BenchmarkResult>(base_config_));
  SingleBenchmark bm{bm_name_, base_config_, std::move(base_executions_), std::move(base_results_)};

  const auto start_test_ts = std::chrono::steady_clock::now();

  bm.generate_data();
  bm.set_up();
  bm.run();

  const BenchmarkResult& result = *bm.get_benchmark_results()[0];

  const std::vector<ExecutionDuration>& op_durations = bm.get_benchmark_results()[0]->total_operation_durations;
  EXPECT_EQ(op_durations.size(), thread_count);
  for (ExecutionDuration duration : op_durations) {
    EXPECT_GT(duration.begin, start_test_ts);
    EXPECT_GT(duration.end, start_test_ts);
    EXPECT_LE(duration.begin, duration.end);
  }

  const uint64_t per_thread_size = TEST_DATA_SIZE / thread_count;
  const std::vector<uint64_t>& op_sizes = bm.get_benchmark_results()[0]->total_operation_sizes;
  EXPECT_EQ(op_sizes.size(), 4);
  EXPECT_EQ(std::accumulate(op_sizes.begin(), op_sizes.end(), 0ul), TEST_DATA_SIZE);
  for (uint64_t size : op_sizes) {
    EXPECT_EQ(size % TEST_CHUNK_SIZE, 0);
  }
}

TEST_F(BenchmarkTest, RunMultiThreadWriteDesc) {
  const size_t num_ops = TEST_DATA_SIZE / 512;
  const size_t thread_count = 16;
  const size_t total_size = 512 * num_ops;
  base_config_.number_threads = thread_count;
  base_config_.access_size = 512;
  base_config_.operation = Operation::Read;
  base_config_.memory_region_size = total_size;

  base_config_.exec_mode = Mode::Sequential_Desc;
  base_executions_.reserve(1);
  base_executions_.push_back(std::make_unique<BenchmarkExecution>());
  base_results_.reserve(1);
  base_results_.push_back(std::make_unique<BenchmarkResult>(base_config_));
  SingleBenchmark bm{bm_name_, base_config_, std::move(base_executions_), std::move(base_results_)};

  const auto start_test_ts = std::chrono::steady_clock::now();

  bm.generate_data();
  bm.set_up();
  bm.run();

  const BenchmarkResult& result = *bm.get_benchmark_results()[0];

  const std::vector<ExecutionDuration>& op_durations = bm.get_benchmark_results()[0]->total_operation_durations;
  EXPECT_EQ(op_durations.size(), thread_count);
  for (ExecutionDuration duration : op_durations) {
    EXPECT_GT(duration.begin, start_test_ts);
    EXPECT_GT(duration.end, start_test_ts);
    EXPECT_LE(duration.begin, duration.end);
  }

  const uint64_t per_thread_size = TEST_DATA_SIZE / thread_count;
  const std::vector<uint64_t>& op_sizes = bm.get_benchmark_results()[0]->total_operation_sizes;
  EXPECT_EQ(op_sizes.size(), 16);
  EXPECT_EQ(std::accumulate(op_sizes.begin(), op_sizes.end(), 0ul), TEST_DATA_SIZE);
  for (uint64_t size : op_sizes) {
    EXPECT_EQ(size % TEST_CHUNK_SIZE, 0);
  }
}

TEST_F(BenchmarkTest, ResultsSingleThreadRead) {
  const size_t num_ops = TEST_DATA_SIZE / 256;
  base_config_.number_threads = 1;
  base_config_.access_size = 256;
  base_config_.operation = Operation::Read;
  base_config_.memory_region_size = TEST_DATA_SIZE;

  BenchmarkResult bm_result{base_config_};

  const uint64_t total_op_duration = 1000000;
  const auto start = std::chrono::steady_clock::now();
  const auto end = start + std::chrono::nanoseconds(total_op_duration);
  bm_result.total_operation_durations.push_back({start, end});
  bm_result.total_operation_sizes.emplace_back(TEST_DATA_SIZE);

  const nlohmann::json& result_json = bm_result.get_result_as_json();
  check_json_result(result_json, TEST_DATA_SIZE, 0.9765625, 1, 0.9765625, 0.0);
}

TEST_F(BenchmarkTest, ResultsSingleThreadWrite) {
  const size_t num_ops = TEST_DATA_SIZE / 256;
  base_config_.number_threads = 1;
  base_config_.access_size = 256;
  base_config_.operation = Operation::Write;
  base_config_.memory_region_size = TEST_DATA_SIZE;

  BenchmarkResult bm_result{base_config_};
  const uint64_t total_op_duration = 2000000;
  const auto start = std::chrono::steady_clock::now();
  const auto end = start + std::chrono::nanoseconds(total_op_duration);
  bm_result.total_operation_durations.push_back({start, end});
  bm_result.total_operation_sizes.emplace_back(TEST_DATA_SIZE);

  const nlohmann::json& result_json = bm_result.get_result_as_json();
  check_json_result(result_json, TEST_DATA_SIZE, 0.48828125, 1, 0.48828125, 0.0);
}

TEST_F(BenchmarkTest, ResultsMultiThreadRead) {
  const size_t num_ops = TEST_DATA_SIZE / 1024;
  const size_t thread_count = 4;
  const size_t num_ops_per_thread = num_ops / thread_count;
  base_config_.number_threads = thread_count;
  base_config_.access_size = 1024;
  base_config_.operation = Operation::Read;
  base_config_.memory_region_size = TEST_DATA_SIZE;

  BenchmarkResult bm_result{base_config_};
  const auto start = std::chrono::steady_clock::now();
  for (size_t thread = 0; thread < thread_count; ++thread) {
    const uint64_t thread_dur = (250000 + (10000 * thread));
    const auto end = start + std::chrono::nanoseconds(thread_dur);
    bm_result.total_operation_durations.push_back({start, end});
    bm_result.total_operation_sizes.emplace_back(TEST_DATA_SIZE / thread_count);
  }

  const nlohmann::json& result_json = bm_result.get_result_as_json();
  check_json_result(result_json, TEST_DATA_SIZE, 3.48772321, 4, 0.8719308, 0.0741378);
}

TEST_F(BenchmarkTest, ResultsMultiThreadWrite) {
  const size_t num_ops = TEST_DATA_SIZE / 512;
  const size_t thread_count = 8;
  const size_t num_ops_per_thread = num_ops / thread_count;
  base_config_.number_threads = thread_count;
  base_config_.access_size = 512;
  base_config_.operation = Operation::Write;
  base_config_.memory_region_size = TEST_DATA_SIZE;

  BenchmarkResult bm_result{base_config_};
  const auto start = std::chrono::steady_clock::now();
  for (size_t thread = 0; thread < thread_count; ++thread) {
    const uint64_t thread_dur = (250000 + (10000 * thread));
    const auto end = start + std::chrono::nanoseconds(thread_dur);
    bm_result.total_operation_durations.push_back({start, end});
    bm_result.total_operation_sizes.emplace_back(TEST_DATA_SIZE / thread_count);
  }

  const nlohmann::json& result_json = bm_result.get_result_as_json();
  check_json_result(result_json, TEST_DATA_SIZE, 3.0517578, 8, 0.38146972, 0.0648887);
}

TEST_F(BenchmarkTest, RunParallelSingleThreadRead) {
  const size_t num_ops = TEST_DATA_SIZE / 256;
  base_config_.number_threads = 1;
  base_config_.access_size = 256;
  base_config_.operation = Operation::Read;
  base_config_.memory_region_size = 256 * num_ops;
  base_config_.min_io_chunk_size = TEST_CHUNK_SIZE;
  base_config_.run_time = 1;

  BenchmarkConfig config_one = base_config_;
  BenchmarkConfig config_two = base_config_;

  config_one.exec_mode = Mode::Sequential;
  config_two.exec_mode = Mode::Random;
  config_two.number_operations = num_ops;

  base_executions_.reserve(2);
  base_executions_.push_back(std::make_unique<BenchmarkExecution>());
  base_executions_.push_back(std::make_unique<BenchmarkExecution>());

  base_results_.reserve(2);
  base_results_.push_back(std::make_unique<BenchmarkResult>(config_one));
  base_results_.push_back(std::make_unique<BenchmarkResult>(config_two));

  ParallelBenchmark bm{
      bm_name_, "sub_bm_1", "sub_bm_2", config_one, config_two, std::move(base_executions_), std::move(base_results_)};
  bm.generate_data();
  bm.set_up();
  bm.run();

  const BenchmarkResult& result_one = *bm.get_benchmark_results()[0];
  const BenchmarkResult& result_two = *bm.get_benchmark_results()[1];

  const std::vector<ExecutionDuration>& all_durations_one = result_one.total_operation_durations;
  const std::vector<ExecutionDuration>& all_durations_two = result_two.total_operation_durations;
  ASSERT_EQ(all_durations_one.size(), 1);
  EXPECT_GT(all_durations_one[0].end - all_durations_one[0].begin, std::chrono::seconds{1});
  ASSERT_EQ(all_durations_two.size(), 1);
  EXPECT_GT(all_durations_two[0].end - all_durations_two[0].begin, std::chrono::seconds{1});

  const std::vector<uint64_t>& all_sizes_one = result_one.total_operation_sizes;
  const std::vector<uint64_t>& all_sizes_two = result_two.total_operation_sizes;
  ASSERT_EQ(all_sizes_one.size(), 1);
  EXPECT_GT(all_sizes_one[0], 0);
  EXPECT_EQ(all_sizes_one[0] % TEST_CHUNK_SIZE, 0);  // can only increase in chunk-sized blocks
  ASSERT_EQ(all_sizes_two.size(), 1);
  EXPECT_GT(all_sizes_two[0], 0);
  EXPECT_EQ(all_sizes_two[0] % TEST_CHUNK_SIZE, 0);
}

TEST_F(BenchmarkTest, ResultsParallelSingleThreadMixed) {
  const size_t num_ops = TEST_DATA_SIZE / 256;
  base_config_.number_threads = 1;
  base_config_.access_size = 256;
  base_config_.memory_region_size = TEST_DATA_SIZE;
  base_config_.min_io_chunk_size = TEST_CHUNK_SIZE;
  base_config_.run_time = 1;

  BenchmarkConfig config_one = base_config_;
  BenchmarkConfig config_two = base_config_;

  config_one.exec_mode = Mode::Sequential;
  config_one.operation = Operation::Write;

  config_two.exec_mode = Mode::Random;
  config_two.operation = Operation::Read;
  config_two.number_operations = num_ops;

  base_executions_.reserve(2);
  base_executions_.push_back(std::make_unique<BenchmarkExecution>());
  base_executions_.push_back(std::make_unique<BenchmarkExecution>());

  base_results_.reserve(2);
  base_results_.push_back(std::make_unique<BenchmarkResult>(config_one));
  base_results_.push_back(std::make_unique<BenchmarkResult>(config_two));

  ParallelBenchmark bm{
      bm_name_, "sub_bm_1", "sub_bm_2", config_one, config_two, std::move(base_executions_), std::move(base_results_)};
  bm.generate_data();
  bm.set_up();
  bm.run();

  const BenchmarkResult& result_one = *bm.get_benchmark_results()[0];
  const BenchmarkResult& result_two = *bm.get_benchmark_results()[1];

  const std::vector<ExecutionDuration>& all_durations_one = result_one.total_operation_durations;
  const std::vector<ExecutionDuration>& all_durations_two = result_two.total_operation_durations;
  ASSERT_EQ(all_durations_one.size(), 1);
  EXPECT_GT(all_durations_one[0].end - all_durations_one[0].begin, std::chrono::seconds{1});
  ASSERT_EQ(all_durations_two.size(), 1);
  EXPECT_GT(all_durations_two[0].end - all_durations_two[0].begin, std::chrono::seconds{1});

  const std::vector<uint64_t>& all_sizes_one = result_one.total_operation_sizes;
  const std::vector<uint64_t>& all_sizes_two = result_two.total_operation_sizes;
  ASSERT_EQ(all_sizes_one.size(), 1);
  EXPECT_GT(all_sizes_one[0], 0);
  EXPECT_EQ(all_sizes_one[0] % TEST_CHUNK_SIZE, 0);  // can only increase in chunk-sized blocks
  ASSERT_EQ(all_sizes_two.size(), 1);
  EXPECT_GT(all_sizes_two[0], 0);
  EXPECT_EQ(all_sizes_two[0] % TEST_CHUNK_SIZE, 0);
}

TEST_F(BenchmarkTest, PrepareDataMemoryNumaLocation) {
  const auto numa_max_node_id = numa_max_node();
  auto* const allowed_memory_nodes_mask = numa_get_mems_allowed();

  for (auto node_id = NumaNodeID{0}; node_id <= numa_max_node_id; ++node_id) {
    if (!numa_bitmask_isbitset(allowed_memory_nodes_mask, node_id)) {
      continue;
    }

    auto config = base_config_;
    config.memory_region_size = 1 * GIB_IN_BYTES;
    config.numa_memory_nodes = NumaNodeIDs{node_id};
    SingleBenchmark bm{bm_name_, config, {}, {}};

    // Generate data creates the memory mapping and populates the memory based on the given numa_memory_nodes.
    bm.generate_data();
    const auto bm_data = bm.get_data()[0];

    const auto region_page_count = config.memory_region_size / utils::PAGE_SIZE;
    for (auto page_idx = uint64_t{0}; page_idx < region_page_count; ++page_idx) {
      auto* addr = bm_data + page_idx * utils::PAGE_SIZE;
      ASSERT_EQ(get_numa_node_index_by_address(addr), node_id);
    }
  }
}

}  // namespace mema
