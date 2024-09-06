#include "benchmark.hpp"

#include <fcntl.h>
#include <numa.h>

#include <fstream>
#include <unordered_set>

#include "gmock/gmock-matchers.h"
#include "gtest/gtest.h"
#include "numa.hpp"
#include "parallel_benchmark.hpp"
#include "single_benchmark.hpp"
#include "test_utils.hpp"
#include "threads.hpp"

namespace mema {

using ::testing::ElementsAre;

constexpr size_t TEST_DATA_SIZE = 1 * MiB;              // 1 MiB
constexpr size_t TEST_BATCH_SIZE = TEST_DATA_SIZE / 8;  // 128 KiB

class BenchmarkTest : public BaseTest {
 protected:
  void SetUp() override {
    base_config_.memory_regions[0].size = TEST_DATA_SIZE;
    base_config_.memory_regions[0].node_ids = {0};
    base_config_.min_io_batch_size = TEST_BATCH_SIZE;
    base_config_.numa_thread_nodes = {0};

    const auto numa_max_node_id = numa_max_node();
    auto* const allowed_memory_nodes_mask = numa_get_mems_allowed();
    valid_node_ids.reserve(numa_max_node_id);

    for (auto node_id = NumaNodeID{0}; node_id <= numa_max_node_id; ++node_id) {
      if (!numa_bitmask_isbitset(allowed_memory_nodes_mask, node_id)) {
        continue;
      }
      valid_node_ids.push_back(node_id);
    }
    valid_node_ids.shrink_to_fit();
  }

  NumaNodeIDs valid_node_ids{};
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

  const std::vector<ThreadConfig>& thread_configs = bm.get_thread_configs()[0];
  ASSERT_EQ(thread_configs.size(), 1);
  const ThreadConfig& thread_config = thread_configs[0];

  EXPECT_EQ(thread_config.thread_idx, 0);
  EXPECT_EQ(thread_config.thread_count, 1);
  EXPECT_EQ(thread_config.primary_region_size, TEST_DATA_SIZE);
  EXPECT_EQ(thread_config.ops_count_per_batch, TEST_BATCH_SIZE / 256);
  EXPECT_EQ(thread_config.batch_count, 8);
  EXPECT_EQ(thread_config.primary_start_addr, bm.get_memory_regions()[0][0]);
  EXPECT_EQ(&thread_config.config, &bm.get_benchmark_configs()[0]);

  const std::vector<ExecutionDuration>& op_durations = bm.get_benchmark_results()[0]->total_operation_durations;
  ASSERT_EQ(op_durations.size(), 1);
  EXPECT_EQ(thread_config.total_operation_duration, &op_durations[0]);

  const std::vector<uint64_t>& op_sizes = bm.get_benchmark_results()[0]->total_operation_sizes;
  ASSERT_EQ(op_sizes.size(), 1);
  EXPECT_EQ(thread_config.total_operation_size, &op_sizes[0]);

  bm.get_benchmark_results()[0]->config.validate();
}

TEST_F(BenchmarkTest, SetUpMultiThread) {
  const size_t thread_count = 4;
  base_config_.number_threads = thread_count;
  base_config_.access_size = 256;

  base_executions_.reserve(1);
  base_executions_.push_back(std::make_unique<BenchmarkExecution>());
  base_results_.reserve(1);
  base_results_.push_back(std::make_unique<BenchmarkResult>(base_config_));
  SingleBenchmark bm{bm_name_, base_config_, std::move(base_executions_), std::move(base_results_)};
  bm.generate_data();
  bm.set_up();

  const size_t region_size = TEST_DATA_SIZE;
  const std::vector<ThreadConfig>& thread_configs = bm.get_thread_configs()[0];
  ASSERT_EQ(thread_configs.size(), thread_count);
  const ThreadConfig& thread_config0 = thread_configs[0];
  const ThreadConfig& thread_config1 = thread_configs[1];
  const ThreadConfig& thread_config2 = thread_configs[2];
  const ThreadConfig& thread_config3 = thread_configs[3];

  EXPECT_EQ(thread_config0.thread_idx, 0);
  EXPECT_EQ(thread_config1.thread_idx, 1);
  EXPECT_EQ(thread_config2.thread_idx, 2);
  EXPECT_EQ(thread_config3.thread_idx, 3);

  EXPECT_EQ(thread_config0.primary_start_addr, bm.get_memory_regions()[0][0]);
  EXPECT_EQ(thread_config1.primary_start_addr, bm.get_memory_regions()[0][0]);
  EXPECT_EQ(thread_config2.primary_start_addr, bm.get_memory_regions()[0][0]);
  EXPECT_EQ(thread_config3.primary_start_addr, bm.get_memory_regions()[0][0]);

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
  for (const ThreadConfig& tc : thread_configs) {
    EXPECT_EQ(tc.thread_count, 4);
    EXPECT_EQ(tc.primary_region_size, region_size);
    EXPECT_EQ(tc.ops_count_per_batch, TEST_BATCH_SIZE / 256);
    EXPECT_EQ(tc.batch_count, 8);
    EXPECT_EQ(&tc.config, &bm.get_benchmark_configs()[0]);
  }
  bm.get_benchmark_results()[0]->config.validate();
}

TEST_F(BenchmarkTest, SetUpSingleThreadCustomOps) {
  // Cumulative size is 960.
  base_config_.custom_operations =
      std::vector<CustomOp>{CustomOp::from_string("m0_r_64"), CustomOp::from_string("m0_r_128"),
                            CustomOp::from_string("m0_r_256"), CustomOp::from_string("m0_r_512")};
  base_config_.number_threads = 1;
  base_config_.exec_mode = Mode::Custom;
  base_config_.number_operations = 10000;

  base_executions_.reserve(1);
  base_executions_.push_back(std::make_unique<BenchmarkExecution>());
  base_results_.reserve(1);
  base_results_.push_back(std::make_unique<BenchmarkResult>(base_config_));
  SingleBenchmark bm{bm_name_, base_config_, std::move(base_executions_), std::move(base_results_)};
  bm.generate_data();
  bm.set_up();

  const std::vector<ThreadConfig>& thread_configs = bm.get_thread_configs()[0];
  ASSERT_EQ(thread_configs.size(), 1);
  const ThreadConfig& thread_config = thread_configs[0];

  EXPECT_EQ(thread_config.thread_idx, 0);
  EXPECT_EQ(thread_config.primary_region_size, TEST_DATA_SIZE);
  EXPECT_EQ(thread_config.ops_count_per_batch, 136 /* = 128 KiB / 960*/);
  EXPECT_EQ(thread_config.batch_count, 74 /* = 10000 / 136 + 1 extra batch */);
  EXPECT_EQ(thread_config.primary_start_addr, bm.get_memory_regions()[0][0]);
  EXPECT_EQ(&thread_config.config, &bm.get_benchmark_configs()[0]);

  const std::vector<ExecutionDuration>& op_durations = bm.get_benchmark_results()[0]->total_operation_durations;
  ASSERT_EQ(op_durations.size(), 1);
  EXPECT_EQ(thread_config.total_operation_duration, &op_durations[0]);

  const std::vector<uint64_t>& op_sizes = bm.get_benchmark_results()[0]->total_operation_sizes;
  ASSERT_EQ(op_sizes.size(), 1);
  EXPECT_EQ(thread_config.total_operation_size, &op_sizes[0]);

  bm.get_benchmark_results()[0]->config.validate();
}

TEST_F(BenchmarkTest, ThreadConfigPinningAllNumaCores) {
  base_config_.numa_thread_nodes = NumaNodeIDs{0};
  auto numa_node_0_cores = core_ids_of_nodes(base_config_.numa_thread_nodes);
  if (numa_node_0_cores.size() < 4) {
    GTEST_SKIP() << "Skipping test: less then 4 cores available for NUMA node 0.";
  }
  base_config_.number_threads = 4;
  base_config_.access_size = 512;
  base_config_.thread_pin_mode = ThreadPinMode::AllNumaCores;

  base_executions_.reserve(1);
  base_executions_.push_back(std::make_unique<BenchmarkExecution>());
  base_results_.reserve(1);
  base_results_.push_back(std::make_unique<BenchmarkResult>(base_config_));
  SingleBenchmark bm{bm_name_, base_config_, std::move(base_executions_), std::move(base_results_)};
  bm.generate_data();
  bm.set_up();

  const auto& thread_configs = bm.get_thread_configs()[0];
  ASSERT_EQ(thread_configs.size(), 4);
  for (auto thread_config_idx = 0u; auto& config : thread_configs) {
    SCOPED_TRACE("Thread config id " + thread_config_idx);
    ASSERT_EQ(config.affinity_core_ids, numa_node_0_cores);
    ++thread_config_idx;
  }
}

TEST_F(BenchmarkTest, ThreadConfigPinningSingleNumaCore) {
  base_config_.numa_thread_nodes = NumaNodeIDs{0};
  auto numa_node_0_cores = core_ids_of_nodes(base_config_.numa_thread_nodes);
  if (numa_node_0_cores.size() < 4) {
    GTEST_SKIP() << "Skipping test: less then 4 cores available for NUMA node 0.";
  }
  base_config_.number_threads = 4;
  base_config_.access_size = 512;
  base_config_.thread_pin_mode = ThreadPinMode::SingleNumaCoreIncrement;

  base_executions_.reserve(1);
  base_executions_.push_back(std::make_unique<BenchmarkExecution>());
  base_results_.reserve(1);
  base_results_.push_back(std::make_unique<BenchmarkResult>(base_config_));
  SingleBenchmark bm{bm_name_, base_config_, std::move(base_executions_), std::move(base_results_)};
  bm.generate_data();
  bm.set_up();

  const auto& thread_configs = bm.get_thread_configs()[0];
  ASSERT_EQ(thread_configs.size(), 4);
  for (auto thread_config_idx = 0u; auto& config : thread_configs) {
    SCOPED_TRACE("Thread config id " + thread_config_idx);
    ASSERT_EQ(config.affinity_core_ids, CoreIDs{numa_node_0_cores[thread_config_idx]});
    ++thread_config_idx;
  }
}

TEST_F(BenchmarkTest, ThreadConfigPinningSingleCore) {
  base_config_.numa_thread_nodes = NumaNodeIDs{0};
  auto numa_node_0_cores = core_ids_of_nodes(base_config_.numa_thread_nodes);
  if (numa_node_0_cores.size() < 4) {
    GTEST_SKIP() << "Skipping test: less then 4 cores available for NUMA node 0.";
  }
  base_config_.number_threads = 4;
  base_config_.access_size = 512;
  base_config_.thread_pin_mode = ThreadPinMode::SingleCoreFixed;
  base_config_.thread_core_ids = CoreIDs{4, 6, 9, 13};

  base_executions_.reserve(1);
  base_executions_.push_back(std::make_unique<BenchmarkExecution>());
  base_results_.reserve(1);
  base_results_.push_back(std::make_unique<BenchmarkResult>(base_config_));
  SingleBenchmark bm{bm_name_, base_config_, std::move(base_executions_), std::move(base_results_)};
  bm.generate_data();
  bm.set_up();

  const auto& thread_configs = bm.get_thread_configs()[0];
  ASSERT_EQ(thread_configs.size(), 4);
  ASSERT_EQ(thread_configs[0].affinity_core_ids, CoreIDs{4});
  ASSERT_EQ(thread_configs[1].affinity_core_ids, CoreIDs{6});
  ASSERT_EQ(thread_configs[2].affinity_core_ids, CoreIDs{9});
  ASSERT_EQ(thread_configs[3].affinity_core_ids, CoreIDs{13});
}

TEST_F(BenchmarkTest, ThreadPinning) {
  auto numa_node_0_cores = core_ids_of_nodes(NumaNodeIDs{0});
  if (numa_node_0_cores.size() < 4) {
    GTEST_SKIP() << "Skipping test: less then 4 cores available for NUMA node 0.";
  }

  const auto core_sets = std::vector<CoreIDs>{
      CoreIDs{numa_node_0_cores[0]},
      CoreIDs{numa_node_0_cores[1]},
      CoreIDs{numa_node_0_cores[2]},
      CoreIDs{numa_node_0_cores[3]},
      CoreIDs{numa_node_0_cores[0], numa_node_0_cores[1]},
      CoreIDs{numa_node_0_cores[0], numa_node_0_cores[1], numa_node_0_cores[2]},
      CoreIDs{numa_node_0_cores[0], numa_node_0_cores[1], numa_node_0_cores[2], numa_node_0_cores[3]}};

  auto work = [&](uint64_t config_idx) {
    auto& core_ids = core_sets[config_idx];
    pin_thread_to_cores(core_ids);
    ASSERT_EQ(core_ids, allowed_thread_core_ids());
  };

  const auto config_count = core_sets.size();
  auto threads = std::vector<std::thread>{};
  threads.reserve(config_count);

  for (auto thread_id = uint64_t{0}; thread_id < config_count; ++thread_id) {
    threads.emplace_back(work, thread_id);
  }

  for (auto& thread : threads) {
    thread.join();
  }
}

TEST_F(BenchmarkTest, RunSingleThreadRead) {
  const size_t num_ops = TEST_DATA_SIZE / 256;
  base_config_.number_threads = 1;
  base_config_.access_size = 256;
  base_config_.operation = Operation::Read;
  base_config_.memory_regions[0].size = 256 * num_ops;

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
  base_config_.memory_regions[0].size = total_size;

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
  base_config_.memory_regions[0].size = 1024 * num_ops;

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
    EXPECT_EQ(size % TEST_BATCH_SIZE, 0);
  }
}

TEST_F(BenchmarkTest, RunMultiThreadWrite) {
  const size_t num_ops = TEST_DATA_SIZE / 512;
  const size_t thread_count = 16;
  const size_t total_size = 512 * num_ops;
  base_config_.number_threads = thread_count;
  base_config_.access_size = 512;
  base_config_.operation = Operation::Read;
  base_config_.memory_regions[0].size = total_size;

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
    EXPECT_EQ(size % TEST_BATCH_SIZE, 0);
  }
}

TEST_F(BenchmarkTest, RunMultiThreadReadDesc) {
  const size_t num_ops = TEST_DATA_SIZE / 1024;
  const size_t thread_count = 4;
  base_config_.number_threads = thread_count;
  base_config_.access_size = 1024;
  base_config_.operation = Operation::Read;
  base_config_.memory_regions[0].size = 1024 * num_ops;

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
    EXPECT_EQ(size % TEST_BATCH_SIZE, 0);
  }
}

TEST_F(BenchmarkTest, RunMultiThreadWriteDesc) {
  const size_t num_ops = TEST_DATA_SIZE / 512;
  const size_t thread_count = 16;
  const size_t total_size = 512 * num_ops;
  base_config_.number_threads = thread_count;
  base_config_.access_size = 512;
  base_config_.operation = Operation::Read;
  base_config_.memory_regions[0].size = total_size;

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
    EXPECT_EQ(size % TEST_BATCH_SIZE, 0);
  }
}

TEST_F(BenchmarkTest, ResultsSingleThreadRead) {
  const size_t num_ops = TEST_DATA_SIZE / 256;
  base_config_.number_threads = 1;
  base_config_.access_size = 256;
  base_config_.operation = Operation::Read;
  base_config_.memory_regions[0].size = TEST_DATA_SIZE;

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
  base_config_.memory_regions[0].size = TEST_DATA_SIZE;

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
  base_config_.memory_regions[0].size = TEST_DATA_SIZE;

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
  base_config_.memory_regions[0].size = TEST_DATA_SIZE;

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
  base_config_.memory_regions[0].size = 256 * num_ops;
  base_config_.min_io_batch_size = TEST_BATCH_SIZE;
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
  EXPECT_EQ(all_sizes_one[0] % TEST_BATCH_SIZE, 0);  // can only increase in batch-sized blocks
  ASSERT_EQ(all_sizes_two.size(), 1);
  EXPECT_GT(all_sizes_two[0], 0);
  EXPECT_EQ(all_sizes_two[0] % TEST_BATCH_SIZE, 0);
}

TEST_F(BenchmarkTest, ResultsParallelSingleThreadMixed) {
  const size_t num_ops = TEST_DATA_SIZE / 256;
  base_config_.number_threads = 1;
  base_config_.access_size = 256;
  base_config_.memory_regions[0].size = TEST_DATA_SIZE;
  base_config_.min_io_batch_size = TEST_BATCH_SIZE;
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
  EXPECT_EQ(all_sizes_one[0] % TEST_BATCH_SIZE, 0);  // can only increase in batch-sized blocks
  ASSERT_EQ(all_sizes_two.size(), 1);
  EXPECT_GT(all_sizes_two[0], 0);
  EXPECT_EQ(all_sizes_two[0] % TEST_BATCH_SIZE, 0);
}

TEST_F(BenchmarkTest, PrepareDataMemoryLocationInterleaved) {
  // Assume two regions. Store for each reagion the previously checked memory nodes.
  auto last_nodes = std::array<std::optional<NumaNodeIDs>, 2>{};
  for (auto node_id : valid_node_ids) {
    SCOPED_TRACE("NUMA node index: " + std::to_string(node_id));

    auto config = base_config_;
    config.memory_regions[0].size = 10 * MiB;
    config.memory_regions[0].node_ids = NumaNodeIDs{node_id};
    config.memory_regions[1].size = 10 * MiB;
    config.memory_regions[1].node_ids = NumaNodeIDs{node_id};
    EXPECT_EQ(config.memory_regions[0].placement_mode(), PagePlacementMode::Interleaved);
    EXPECT_EQ(config.memory_regions[1].placement_mode(), PagePlacementMode::Interleaved);

    SingleBenchmark bm{bm_name_, config, {}, {}};

    // Generate data creates the memory mapping and populates the memory based on the given numa_memory_nodes.
    bm.generate_data();

    const auto& regions = bm.get_memory_regions()[0];
    ASSERT_EQ(regions.size(), 2u);
    for (auto region_idx = uint64_t{0}; auto& region : regions) {
      const auto definition = config.memory_regions[region_idx];
      const auto region_page_count = config.memory_regions[region_idx].size / utils::PAGE_SIZE;
      ASSERT_TRUE(verify_interleaved_page_placement(region, definition.size, definition.node_ids));
      if (last_nodes[region_idx]) {
        ASSERT_FALSE(verify_interleaved_page_placement(region, definition.size, *last_nodes[region_idx]));
      }
      last_nodes[region_idx] = definition.node_ids;
      ++region_idx;
    }
  }
}

TEST_F(BenchmarkTest, PrepareDataMemoryLocationPartitioned2Nodes) {
  if (valid_node_ids.size() < 2) {
    GTEST_SKIP() << "Skipping test: system has " << valid_node_ids.size() << " but test requires at least 2.";
  }

  auto config = base_config_;
  config.memory_regions[0].size = 10 * MiB;
  config.memory_regions[0].node_ids = NumaNodeIDs{valid_node_ids[0], valid_node_ids[1]};
  config.memory_regions[0].percentage_pages_first_partition = 60;
  config.memory_regions[0].node_count_first_partition = 1;
  EXPECT_EQ(config.memory_regions[0].placement_mode(), PagePlacementMode::Partitioned);
  config.memory_regions[1].size = 0;
  SingleBenchmark bm{bm_name_, config, {}, {}};
  // Generate data creates the memory mapping and populates the memory based on the given numa_memory_nodes.
  bm.generate_data();

  const auto& regions = bm.get_memory_regions()[0];
  ASSERT_EQ(regions.size(), 2u);
  const auto& definition = config.memory_regions[0];
  const auto region_page_count = definition.size / utils::PAGE_SIZE;
  // Verify page status
  auto page_status = retrieve_page_status(region_page_count, regions[0]);
  auto page_idx = uint64_t{0};
  // Partition 1
  const auto first_partition_page_count =
      static_cast<uint32_t>((*definition.percentage_pages_first_partition / 100.f) * region_page_count);
  for (; page_idx < first_partition_page_count; ++page_idx) {
    ASSERT_EQ(page_status[page_idx], valid_node_ids[0]);
  }
  // Partition 2
  for (; page_idx < region_page_count; ++page_idx) {
    ASSERT_EQ(page_status[page_idx], valid_node_ids[1]);
  }

  ASSERT_TRUE(verify_partitioned_page_placement(regions[0], definition.size, definition.node_ids,
                                                *definition.percentage_pages_first_partition,
                                                *definition.node_count_first_partition));
  ASSERT_FALSE(verify_interleaved_page_placement(regions[0], definition.size, definition.node_ids));

  ASSERT_EQ(regions[1], nullptr);
}

TEST_F(BenchmarkTest, PrepareDataMemoryLocationPartitioned3Nodes) {
  if (valid_node_ids.size() < 2) {
    GTEST_SKIP() << "Skipping test: system has " << valid_node_ids.size() << " but test requires at least 2.";
  }

  auto config = base_config_;
  config.memory_regions[0].size = 10 * MiB;
  config.memory_regions[0].node_ids = NumaNodeIDs{valid_node_ids[0], valid_node_ids[1], valid_node_ids[1]};
  config.memory_regions[0].percentage_pages_first_partition = 60;
  config.memory_regions[0].node_count_first_partition = 2;
  EXPECT_EQ(config.memory_regions[0].placement_mode(), PagePlacementMode::Partitioned);
  config.memory_regions[1].size = 0;
  SingleBenchmark bm{bm_name_, config, {}, {}};
  // Generate data creates the memory mapping and populates the memory based on the given numa_memory_nodes.
  bm.generate_data();

  const auto& regions = bm.get_memory_regions()[0];
  ASSERT_EQ(regions.size(), 2u);
  const auto& definition = config.memory_regions[0];
  const auto region_page_count = definition.size / utils::PAGE_SIZE;
  // Verify page status
  auto page_status = retrieve_page_status(region_page_count, regions[0]);
  auto page_idx = uint64_t{0};
  // Partition 1
  const auto first_partition_nodes = std::unordered_set{valid_node_ids[0], valid_node_ids[1]};
  const auto first_partition_page_count =
      static_cast<uint32_t>((*definition.percentage_pages_first_partition / 100.f) * region_page_count);
  auto last_status = std::optional<uint64_t>(std::nullopt);
  for (; page_idx < first_partition_page_count; ++page_idx) {
    const auto& status = page_status[page_idx];
    ASSERT_TRUE(first_partition_nodes.contains(status));
    if (last_status) {
      ASSERT_NE(*last_status, status);
    }
    last_status = status;
  }
  // Partition 2
  for (; page_idx < region_page_count; ++page_idx) {
    ASSERT_EQ(page_status[page_idx], valid_node_ids[1]);
  }

  ASSERT_TRUE(verify_partitioned_page_placement(regions[0], definition.size, definition.node_ids,
                                                *definition.percentage_pages_first_partition,
                                                *definition.node_count_first_partition));
  ASSERT_FALSE(verify_interleaved_page_placement(regions[0], definition.size, definition.node_ids));

  ASSERT_EQ(regions[1], nullptr);
}

TEST_F(BenchmarkTest, PrepareDataMemoryLocationPartitioned4Nodes) {
  if (valid_node_ids.size() < 2) {
    GTEST_SKIP() << "Skipping test: system has " << valid_node_ids.size() << " but test requires at least 2.";
  }

  auto config = base_config_;
  config.memory_regions[0].size = 10 * MiB;
  config.memory_regions[0].node_ids =
      NumaNodeIDs{valid_node_ids[0], valid_node_ids[1], valid_node_ids[1], valid_node_ids[0]};
  config.memory_regions[0].percentage_pages_first_partition = 60;
  config.memory_regions[0].node_count_first_partition = 2;
  EXPECT_EQ(config.memory_regions[0].placement_mode(), PagePlacementMode::Partitioned);
  config.memory_regions[1].size = 0;
  SingleBenchmark bm{bm_name_, config, {}, {}};
  // Generate data creates the memory mapping and populates the memory based on the given numa_memory_nodes.
  bm.generate_data();

  const auto& regions = bm.get_memory_regions()[0];
  ASSERT_EQ(regions.size(), 2u);
  const auto& definition = config.memory_regions[0];
  const auto region_page_count = definition.size / utils::PAGE_SIZE;
  // Verify page status
  auto page_status = retrieve_page_status(region_page_count, regions[0]);
  auto page_idx = uint64_t{0};
  // Partition 1
  const auto first_partition_nodes = std::unordered_set{valid_node_ids[0], valid_node_ids[1]};
  const auto first_partition_page_count =
      static_cast<uint32_t>((*definition.percentage_pages_first_partition / 100.f) * region_page_count);
  auto last_status = std::optional<uint64_t>(std::nullopt);
  for (; page_idx < first_partition_page_count; ++page_idx) {
    const auto& status = page_status[page_idx];
    ASSERT_TRUE(first_partition_nodes.contains(status));
    if (last_status) {
      ASSERT_NE(*last_status, status);
    }
    last_status = status;
  }
  // Partition 2
  const auto second_partition_nodes = std::unordered_set{valid_node_ids[0], valid_node_ids[1]};
  last_status = std::optional<uint64_t>(std::nullopt);
  for (; page_idx < region_page_count; ++page_idx) {
    const auto& status = page_status[page_idx];
    ASSERT_TRUE(second_partition_nodes.contains(status));
    if (last_status) {
      ASSERT_NE(*last_status, status);
    }
    last_status = status;
  }

  ASSERT_TRUE(verify_partitioned_page_placement(regions[0], definition.size, definition.node_ids,
                                                *definition.percentage_pages_first_partition,
                                                *definition.node_count_first_partition));
  ASSERT_FALSE(verify_interleaved_page_placement(regions[0], definition.size, definition.node_ids));

  ASSERT_EQ(regions[1], nullptr);
}

TEST_F(BenchmarkTest, PrepareDataMemoryLocationInterleavedPartitioned) {
  if (valid_node_ids.size() < 2) {
    GTEST_SKIP() << "Skipping test: system has " << valid_node_ids.size() << " but test requires at least 2.";
  }

  const auto percentage_first_node = 60u;
  auto config = base_config_;
  config.memory_regions[0].size = 10 * MiB;
  config.memory_regions[0].node_ids = NumaNodeIDs{valid_node_ids[0], valid_node_ids[1]};
  ASSERT_EQ(config.memory_regions[0].placement_mode(), PagePlacementMode::Interleaved);

  config.memory_regions[1].size = 20 * MiB;
  config.memory_regions[1].node_ids = NumaNodeIDs{valid_node_ids[0], valid_node_ids[1]};
  config.memory_regions[1].percentage_pages_first_partition = 60;
  config.memory_regions[1].node_count_first_partition = 1;
  ASSERT_EQ(config.memory_regions[1].placement_mode(), PagePlacementMode::Partitioned);

  SingleBenchmark bm{bm_name_, config, {}, {}};

  // Generate data creates the memory mapping and populates the memory based on the given numa_memory_nodes.
  bm.generate_data();

  const auto& regions = bm.get_memory_regions()[0];
  ASSERT_EQ(regions.size(), 2u);

  const auto& definitions = config.memory_regions;
  // Region 0
  ASSERT_FALSE(verify_partitioned_page_placement(regions[0], definitions[0].size, definitions[0].node_ids,
                                                 percentage_first_node, 1));
  ASSERT_TRUE(verify_interleaved_page_placement(regions[0], definitions[0].size, definitions[0].node_ids));
  // Region 1
  ASSERT_TRUE(verify_partitioned_page_placement(regions[1], definitions[1].size, definitions[1].node_ids,
                                                percentage_first_node, 1));
  ASSERT_FALSE(verify_interleaved_page_placement(regions[1], definitions[1].size, definitions[1].node_ids));
}

}  // namespace mema
