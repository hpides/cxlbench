#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/spdlog.h>
#include <yaml-cpp/yaml.h>

#include <filesystem>
#include <fstream>

#include "benchmark_config.hpp"
#include "benchmark_factory.hpp"
#include "gtest/gtest.h"
#include "test_utils.hpp"

namespace mema {

constexpr auto TEST_SINGLE_MATRIX = "test_single_matrix.yaml";
constexpr auto TEST_SINGLE_SEQUENTIAL = "test_single_sequential.yaml";
constexpr auto TEST_SINGLE_RANDOM = "test_single_random.yaml";
constexpr auto TEST_PARALLEL_SEQUENTIAL_RANDOM = "test_parallel_sequential_random.yaml";
constexpr auto TEST_PARALLEL_MATRIX = "test_parallel_matrix.yaml";
constexpr auto TEST_CUSTOM_OPERATIONS_MATRIX = "test_custom_operations_matrix.yaml";
constexpr auto TEST_INVALID_NUMA_MEMORY_NODES = "test_invalid_numa_memory_nodes.yaml";
constexpr auto TEST_INVALID_NUMA_TASK_NODES = "test_invalid_numa_task_nodes.yaml";
constexpr auto TEST_THREAD_MULTI_CORE_NUMA = "test_thread_multi_core_numa.yaml";
constexpr auto TEST_THREAD_SINGLE_CORE_NUMA = "test_thread_single_core_numa.yaml";
constexpr auto TEST_THREAD_SINGLE_CORE = "test_thread_single_core.yaml";

class ConfigTest : public BaseTest {
 protected:
  static void SetUpTestSuite() {
    test_logger_path = std::filesystem::temp_directory_path() / "test-logger.log";
    auto file_logger = spdlog::basic_logger_mt("test-logger", test_logger_path.string());
    spdlog::set_default_logger(file_logger);
  }

  static void TearDownTestSuite() { std::filesystem::remove(test_logger_path); }

  void SetUp() override {
    const std::filesystem::path test_config_path = std::filesystem::current_path() / "resources" / "configs";
    config_single_matrix = BenchmarkFactory::get_config_files(test_config_path / TEST_SINGLE_MATRIX);
    config_single_sequential = BenchmarkFactory::get_config_files(test_config_path / TEST_SINGLE_SEQUENTIAL);
    config_single_random = BenchmarkFactory::get_config_files(test_config_path / TEST_SINGLE_RANDOM);
    config_parallel_sequential_random =
        BenchmarkFactory::get_config_files(test_config_path / TEST_PARALLEL_SEQUENTIAL_RANDOM);
    config_parallel_matrix = BenchmarkFactory::get_config_files(test_config_path / TEST_PARALLEL_MATRIX);
    config_custom_operations_matrix =
        BenchmarkFactory::get_config_files(test_config_path / TEST_CUSTOM_OPERATIONS_MATRIX);
    config_invalid_numa_memory_nodes =
        BenchmarkFactory::get_config_files(test_config_path / TEST_INVALID_NUMA_MEMORY_NODES);
    config_invalid_numa_task_nodes =
        BenchmarkFactory::get_config_files(test_config_path / TEST_INVALID_NUMA_TASK_NODES);
    config_thread_multi_core_numa = BenchmarkFactory::get_config_files(test_config_path / TEST_THREAD_MULTI_CORE_NUMA);
    config_thread_single_core_numa =
        BenchmarkFactory::get_config_files(test_config_path / TEST_THREAD_SINGLE_CORE_NUMA);
    config_thread_single_core = BenchmarkFactory::get_config_files(test_config_path / TEST_THREAD_SINGLE_CORE);

    // Set required NUMA nodes.
    bm_config.memory_regions[0].node_ids = {42};
    bm_config.memory_regions[1].node_ids = {1337};
    bm_config.numa_thread_nodes = {0};
  }

  void TearDown() override { std::ofstream empty_log(test_logger_path, std::ostream::trunc); }

  static void check_log_for_critical(const std::string& expected_msg) {
    // Make sure content is written to the log file
    spdlog::default_logger()->flush();

    std::stringstream raw_log_content;
    std::ifstream log_checker(test_logger_path);
    raw_log_content << log_checker.rdbuf();
    std::string log_content = raw_log_content.str();

    if (log_content.find("critical") == std::string::npos) {
      FAIL() << "Did not find keyword 'critical' in log file.";
    }

    if (log_content.find(expected_msg) == std::string::npos) {
      FAIL() << "Did not find expected '" << expected_msg << "' in log file.";
    }
  }

  std::vector<YAML::Node> config_single_matrix;
  std::vector<YAML::Node> config_single_sequential;
  std::vector<YAML::Node> config_single_random;
  std::vector<YAML::Node> config_parallel_sequential_random;
  std::vector<YAML::Node> config_parallel_matrix;
  std::vector<YAML::Node> config_custom_operations_matrix;
  std::vector<YAML::Node> config_invalid_numa_memory_nodes;
  std::vector<YAML::Node> config_invalid_numa_task_nodes;
  std::vector<YAML::Node> config_thread_multi_core_numa;
  std::vector<YAML::Node> config_thread_single_core_numa;
  std::vector<YAML::Node> config_thread_single_core;
  BenchmarkConfig bm_config;
  static std::filesystem::path test_logger_path;
};

std::filesystem::path ConfigTest::test_logger_path;

TEST_F(ConfigTest, PlacementMode) {
  bm_config.memory_regions[0].percentage_pages_first_partition = 42;
  bm_config.memory_regions[1].percentage_pages_first_partition = std::nullopt;
  EXPECT_EQ(bm_config.memory_regions[0].placement_mode(), PagePlacementMode::Partitioned);
  EXPECT_EQ(bm_config.memory_regions[1].placement_mode(), PagePlacementMode::Interleaved);
}

TEST_F(ConfigTest, SingleDecodeSequential) {
  std::vector<SingleBenchmark> single_bms = BenchmarkFactory::create_single_benchmarks(config_single_sequential);
  std::vector<ParallelBenchmark> parallel_benchmarks =
      BenchmarkFactory::create_parallel_benchmarks(config_single_sequential);
  ASSERT_EQ(single_bms.size(), 1);
  ASSERT_EQ(parallel_benchmarks.size(), 0);
  bm_config = single_bms.at(0).get_benchmark_configs()[0];

  BenchmarkConfig bm_config_default{};

  EXPECT_EQ(bm_config.memory_regions[0].size, 67108864);
  EXPECT_EQ(bm_config.access_size, 256);
  EXPECT_EQ(bm_config.exec_mode, Mode::Sequential);

  EXPECT_EQ(bm_config.number_threads, 2);

  EXPECT_EQ(bm_config.operation, Operation::Read);
  EXPECT_EQ(bm_config.min_io_batch_size, 16 * 1024);

  EXPECT_EQ(bm_config.number_operations, bm_config_default.number_operations);
  EXPECT_EQ(bm_config.random_distribution, bm_config_default.random_distribution);
  EXPECT_EQ(bm_config.zipf_alpha, bm_config_default.zipf_alpha);
  EXPECT_EQ(bm_config.cache_instruction, bm_config_default.cache_instruction);
  EXPECT_EQ(bm_config.run_time, bm_config_default.run_time);
  EXPECT_EQ(bm_config.latency_sample_frequency, bm_config_default.latency_sample_frequency);
  EXPECT_EQ(bm_config.numa_thread_nodes, NumaNodeIDs{0});
  EXPECT_EQ(bm_config.memory_regions[0].node_ids, (NumaNodeIDs{0, 1}));
  EXPECT_EQ(bm_config.memory_regions[0].percentage_pages_first_partition, 37);
  EXPECT_EQ(bm_config.memory_regions[0].node_count_first_partition, 1);
}

TEST_F(ConfigTest, DecodeRandom) {
  std::vector<SingleBenchmark> single_bms = BenchmarkFactory::create_single_benchmarks(config_single_random);
  ASSERT_EQ(single_bms.size(), 1);
  bm_config = single_bms.at(0).get_benchmark_configs()[0];

  BenchmarkConfig bm_config_default{};

  EXPECT_EQ(bm_config.exec_mode, Mode::Random);

  EXPECT_EQ(bm_config.random_distribution, RandomDistribution::Zipf);
  EXPECT_EQ(bm_config.zipf_alpha, 0.9);

  EXPECT_EQ(bm_config.operation, Operation::Write);

  EXPECT_EQ(bm_config.memory_regions[0].size, bm_config_default.memory_regions[0].size);
  EXPECT_EQ(bm_config.memory_regions[0].node_ids, NumaNodeIDs{1});
  EXPECT_EQ(bm_config.memory_regions[0].percentage_pages_first_partition,
            bm_config_default.memory_regions[0].percentage_pages_first_partition);
  EXPECT_EQ(bm_config.numa_thread_nodes, NumaNodeIDs{0});
  EXPECT_EQ(bm_config.access_size, bm_config_default.access_size);
  EXPECT_EQ(bm_config.number_operations, bm_config_default.number_operations);
  EXPECT_EQ(bm_config.cache_instruction, bm_config_default.cache_instruction);
  EXPECT_EQ(bm_config.number_threads, bm_config_default.number_threads);
  EXPECT_EQ(bm_config.min_io_batch_size, bm_config_default.min_io_batch_size);
  EXPECT_EQ(bm_config.run_time, bm_config_default.run_time);
  EXPECT_EQ(bm_config.latency_sample_frequency, bm_config_default.latency_sample_frequency);
}

TEST_F(ConfigTest, ParallelDecodeSequentialRandom) {
  std::vector<SingleBenchmark> single_bms =
      BenchmarkFactory::create_single_benchmarks(config_parallel_sequential_random);
  std::vector<ParallelBenchmark> parallel_bms =
      BenchmarkFactory::create_parallel_benchmarks(config_parallel_sequential_random);
  ASSERT_EQ(single_bms.size(), 0);
  ASSERT_EQ(parallel_bms.size(), 1);
  bm_config = parallel_bms.at(0).get_benchmark_configs()[0];

  EXPECT_EQ(parallel_bms.at(0).get_benchmark_name_one(), "buffer_read");
  EXPECT_EQ(parallel_bms.at(0).get_benchmark_name_two(), "logging");

  BenchmarkConfig bm_config_default{};

  EXPECT_EQ(bm_config.memory_regions[0].size, 10737418240);
  EXPECT_EQ(bm_config.memory_regions[0].node_ids, (NumaNodeIDs{0}));
  EXPECT_EQ(bm_config.numa_thread_nodes, (NumaNodeIDs{0, 1}));
  EXPECT_EQ(bm_config.access_size, 4096);
  EXPECT_EQ(bm_config.exec_mode, Mode::Random);
  EXPECT_EQ(bm_config.number_operations, 10000000);

  EXPECT_EQ(bm_config.number_threads, 8);

  EXPECT_EQ(bm_config.operation, Operation::Read);

  EXPECT_EQ(bm_config.random_distribution, bm_config_default.random_distribution);
  EXPECT_EQ(bm_config.zipf_alpha, bm_config_default.zipf_alpha);
  EXPECT_EQ(bm_config.cache_instruction, bm_config_default.cache_instruction);
  EXPECT_EQ(bm_config.run_time, bm_config_default.run_time);
  EXPECT_EQ(bm_config.latency_sample_frequency, bm_config_default.latency_sample_frequency);

  bm_config = parallel_bms.at(0).get_benchmark_configs()[1];

  EXPECT_EQ(bm_config.memory_regions[0].size, 10737418240);
  EXPECT_EQ(bm_config.memory_regions[0].node_ids, (NumaNodeIDs{2, 3}));
  EXPECT_EQ(bm_config.numa_thread_nodes, (NumaNodeIDs{0, 1}));
  EXPECT_EQ(bm_config.access_size, 256);
  EXPECT_EQ(bm_config.exec_mode, Mode::Sequential);
  EXPECT_EQ(bm_config.cache_instruction, CacheInstruction::None);

  EXPECT_EQ(bm_config.number_threads, 16);

  EXPECT_EQ(bm_config.operation, Operation::Write);

  EXPECT_EQ(bm_config.number_operations, bm_config_default.number_operations);
  EXPECT_EQ(bm_config.random_distribution, bm_config_default.random_distribution);
  EXPECT_EQ(bm_config.zipf_alpha, bm_config_default.zipf_alpha);
  EXPECT_EQ(bm_config.min_io_batch_size, bm_config_default.min_io_batch_size);
  EXPECT_EQ(bm_config.latency_sample_frequency, bm_config_default.latency_sample_frequency);
  EXPECT_EQ(bm_config.run_time, bm_config_default.run_time);
  EXPECT_EQ(bm_config.memory_regions[0].percentage_pages_first_partition,
            bm_config_default.memory_regions[0].percentage_pages_first_partition);
}

TEST_F(ConfigTest, DecodeMatrix) {
  const size_t bm_count = 6;
  std::vector<SingleBenchmark> single_bms = BenchmarkFactory::create_single_benchmarks(config_single_matrix);
  std::vector<ParallelBenchmark> parallel_bms = BenchmarkFactory::create_parallel_benchmarks(config_single_matrix);
  ASSERT_EQ(single_bms.size(), bm_count);
  ASSERT_EQ(parallel_bms.size(), 0);
  EXPECT_EQ(single_bms[0].get_benchmark_configs()[0].number_threads, 1);
  EXPECT_EQ(single_bms[0].get_benchmark_configs()[0].access_size, 256);
  EXPECT_EQ(single_bms[1].get_benchmark_configs()[0].number_threads, 1);
  EXPECT_EQ(single_bms[1].get_benchmark_configs()[0].access_size, 4096);
  EXPECT_EQ(single_bms[2].get_benchmark_configs()[0].number_threads, 2);
  EXPECT_EQ(single_bms[2].get_benchmark_configs()[0].access_size, 256);
  EXPECT_EQ(single_bms[3].get_benchmark_configs()[0].number_threads, 2);
  EXPECT_EQ(single_bms[3].get_benchmark_configs()[0].access_size, 4096);
  EXPECT_EQ(single_bms[4].get_benchmark_configs()[0].number_threads, 4);
  EXPECT_EQ(single_bms[4].get_benchmark_configs()[0].access_size, 256);
  EXPECT_EQ(single_bms[5].get_benchmark_configs()[0].number_threads, 4);
  EXPECT_EQ(single_bms[5].get_benchmark_configs()[0].access_size, 4096);

  BenchmarkConfig bm_config_default{};
  for (size_t bm_index = 0; bm_index < bm_count; ++bm_index) {
    const SingleBenchmark& bm = single_bms[bm_index];
    const BenchmarkConfig& config = bm.get_benchmark_configs()[0];

    // Other args are identical for all configs
    EXPECT_EQ(config.memory_regions[0].size, 536870912);
    EXPECT_EQ(config.memory_regions[0].node_ids, (NumaNodeIDs{0, 3}));
    EXPECT_EQ(config.exec_mode, Mode::Sequential);
    EXPECT_EQ(config.operation, Operation::Read);
    EXPECT_EQ(config.numa_thread_nodes, NumaNodeIDs{0});

    EXPECT_EQ(config.number_operations, bm_config_default.number_operations);
    EXPECT_EQ(config.random_distribution, bm_config_default.random_distribution);
    EXPECT_EQ(config.zipf_alpha, bm_config_default.zipf_alpha);
    EXPECT_EQ(config.cache_instruction, bm_config_default.cache_instruction);
    EXPECT_EQ(config.min_io_batch_size, bm_config_default.min_io_batch_size);
    EXPECT_EQ(config.run_time, bm_config_default.run_time);
    EXPECT_EQ(config.latency_sample_frequency, bm_config_default.latency_sample_frequency);
    EXPECT_EQ(config.memory_regions[0].percentage_pages_first_partition,
              bm_config_default.memory_regions[0].percentage_pages_first_partition);
  }
}

TEST_F(ConfigTest, DecodeCustomOperationsMatrix) {
  const size_t bm_count = 3;
  std::vector<SingleBenchmark> single_bms = BenchmarkFactory::create_single_benchmarks(config_custom_operations_matrix);

  ASSERT_EQ(single_bms.size(), bm_count);
  EXPECT_EQ(single_bms[0].get_benchmark_configs()[0].custom_operations.size(), 3);
  EXPECT_EQ(single_bms[0].get_benchmark_configs()[0].custom_operations[0].memory_type, MemoryType::Primary);
  EXPECT_EQ(single_bms[0].get_benchmark_configs()[0].custom_operations[0].type, Operation::Read);
  EXPECT_EQ(single_bms[0].get_benchmark_configs()[0].custom_operations[0].size, 64);
  EXPECT_EQ(single_bms[0].get_benchmark_configs()[0].custom_operations[1].memory_type, MemoryType::Primary);
  EXPECT_EQ(single_bms[0].get_benchmark_configs()[0].custom_operations[1].type, Operation::Write);
  EXPECT_EQ(single_bms[0].get_benchmark_configs()[0].custom_operations[1].size, 64);
  EXPECT_EQ(single_bms[0].get_benchmark_configs()[0].custom_operations[1].cache_fn, CacheInstruction::NoCache);
  EXPECT_EQ(single_bms[0].get_benchmark_configs()[0].custom_operations[2].memory_type, MemoryType::Secondary);
  EXPECT_EQ(single_bms[0].get_benchmark_configs()[0].custom_operations[2].type, Operation::Read);
  EXPECT_EQ(single_bms[0].get_benchmark_configs()[0].custom_operations[2].size, 128);

  EXPECT_EQ(single_bms[1].get_benchmark_configs()[0].custom_operations.size(), 2);
  EXPECT_EQ(single_bms[1].get_benchmark_configs()[0].custom_operations[0].memory_type, MemoryType::Primary);
  EXPECT_EQ(single_bms[1].get_benchmark_configs()[0].custom_operations[0].type, Operation::Read);
  EXPECT_EQ(single_bms[1].get_benchmark_configs()[0].custom_operations[0].size, 256);
  EXPECT_EQ(single_bms[1].get_benchmark_configs()[0].custom_operations[1].memory_type, MemoryType::Primary);
  EXPECT_EQ(single_bms[1].get_benchmark_configs()[0].custom_operations[1].type, Operation::Write);
  EXPECT_EQ(single_bms[1].get_benchmark_configs()[0].custom_operations[1].size, 256);
  EXPECT_EQ(single_bms[1].get_benchmark_configs()[0].custom_operations[1].cache_fn, CacheInstruction::None);

  EXPECT_EQ(single_bms[2].get_benchmark_configs()[0].custom_operations.size(), 2);
  EXPECT_EQ(single_bms[2].get_benchmark_configs()[0].custom_operations[0].memory_type, MemoryType::Secondary);
  EXPECT_EQ(single_bms[2].get_benchmark_configs()[0].custom_operations[0].type, Operation::Read);
  EXPECT_EQ(single_bms[2].get_benchmark_configs()[0].custom_operations[0].size, 1024);
  EXPECT_EQ(single_bms[2].get_benchmark_configs()[0].custom_operations[1].memory_type, MemoryType::Secondary);
  EXPECT_EQ(single_bms[2].get_benchmark_configs()[0].custom_operations[1].type, Operation::Write);
  EXPECT_EQ(single_bms[2].get_benchmark_configs()[0].custom_operations[1].size, 128);
  EXPECT_EQ(single_bms[2].get_benchmark_configs()[0].custom_operations[1].cache_fn, CacheInstruction::Cache);

  BenchmarkConfig bm_config_default{};
  for (size_t bm_idx = 0; bm_idx < bm_count; ++bm_idx) {
    const SingleBenchmark& bm = single_bms[bm_idx];
    const BenchmarkConfig& config = bm.get_benchmark_configs()[0];

    // Other args are identical for all configs
    EXPECT_EQ(config.memory_regions[0].size, 2147483648);
    EXPECT_EQ(config.memory_regions[1].size, 1073741824);
    EXPECT_EQ(config.memory_regions[0].node_ids, NumaNodeIDs{1});
    EXPECT_EQ(config.memory_regions[1].node_ids, NumaNodeIDs{2});
    EXPECT_EQ(config.exec_mode, Mode::Custom);
    EXPECT_EQ(config.number_operations, 100000000);
    EXPECT_EQ(config.number_threads, 16);
    EXPECT_EQ(config.numa_thread_nodes, NumaNodeIDs{0});

    EXPECT_EQ(config.random_distribution, bm_config_default.random_distribution);
    EXPECT_EQ(config.zipf_alpha, bm_config_default.zipf_alpha);
    EXPECT_EQ(config.min_io_batch_size, bm_config_default.min_io_batch_size);
    EXPECT_EQ(config.run_time, bm_config_default.run_time);
    EXPECT_EQ(config.latency_sample_frequency, bm_config_default.latency_sample_frequency);
    EXPECT_EQ(config.memory_regions[0].percentage_pages_first_partition,
              bm_config_default.memory_regions[0].percentage_pages_first_partition);
  }
}

TEST_F(ConfigTest, ParallelDecodeMatrix) {
  const uint8_t bm_count = 4;
  std::vector<SingleBenchmark> single_bms = BenchmarkFactory::create_single_benchmarks(config_parallel_matrix);
  std::vector<ParallelBenchmark> parallel_bms = BenchmarkFactory::create_parallel_benchmarks(config_parallel_matrix);
  ASSERT_EQ(single_bms.size(), 0);
  ASSERT_EQ(parallel_bms.size(), bm_count);

  EXPECT_EQ(parallel_bms.at(0).get_benchmark_name_one(), "buffer_read");
  EXPECT_EQ(parallel_bms.at(0).get_benchmark_name_two(), "logging");

  EXPECT_EQ(parallel_bms[0].get_benchmark_configs()[0].number_threads, 8);
  EXPECT_EQ(parallel_bms[0].get_benchmark_configs()[1].access_size, 64);
  EXPECT_EQ(parallel_bms[1].get_benchmark_configs()[0].number_threads, 16);
  EXPECT_EQ(parallel_bms[1].get_benchmark_configs()[1].access_size, 64);
  EXPECT_EQ(parallel_bms[2].get_benchmark_configs()[0].number_threads, 8);
  EXPECT_EQ(parallel_bms[2].get_benchmark_configs()[1].access_size, 256);
  EXPECT_EQ(parallel_bms[3].get_benchmark_configs()[0].number_threads, 16);
  EXPECT_EQ(parallel_bms[3].get_benchmark_configs()[1].access_size, 256);

  BenchmarkConfig bm_config_default{};
  for (size_t bm_idx = 0; bm_idx < bm_count; ++bm_idx) {
    const ParallelBenchmark& bm = parallel_bms[bm_idx];
    const BenchmarkConfig& config_one = bm.get_benchmark_configs()[0];
    const BenchmarkConfig& config_two = bm.get_benchmark_configs()[1];

    // Other args are identical for all configs
    EXPECT_EQ(config_one.memory_regions[0].size, 10737418240);
    EXPECT_EQ(config_one.memory_regions[0].node_ids, NumaNodeIDs{1});
    EXPECT_EQ(config_one.access_size, 4096);
    EXPECT_EQ(config_one.exec_mode, Mode::Random);
    EXPECT_EQ(config_one.number_operations, 10000000);
    EXPECT_EQ(config_one.operation, Operation::Read);
    EXPECT_EQ(config_one.numa_thread_nodes, NumaNodeIDs{0});

    EXPECT_EQ(config_one.random_distribution, bm_config_default.random_distribution);
    EXPECT_EQ(config_one.zipf_alpha, bm_config_default.zipf_alpha);
    EXPECT_EQ(config_one.cache_instruction, bm_config_default.cache_instruction);
    EXPECT_EQ(config_one.min_io_batch_size, bm_config_default.min_io_batch_size);
    EXPECT_EQ(config_one.run_time, bm_config_default.run_time);
    EXPECT_EQ(config_one.latency_sample_frequency, bm_config_default.latency_sample_frequency);

    EXPECT_EQ(config_two.memory_regions[0].size, 10737418240);
    EXPECT_EQ(config_two.memory_regions[0].node_ids, NumaNodeIDs{1});
    EXPECT_EQ(config_two.exec_mode, Mode::Sequential);
    EXPECT_EQ(config_two.operation, Operation::Write);
    EXPECT_EQ(config_two.number_threads, 16);
    EXPECT_EQ(config_two.cache_instruction, CacheInstruction::None);
    EXPECT_EQ(config_two.numa_thread_nodes, NumaNodeIDs{0});

    EXPECT_EQ(config_two.number_operations, bm_config_default.number_operations);
    EXPECT_EQ(config_two.random_distribution, bm_config_default.random_distribution);
    EXPECT_EQ(config_two.zipf_alpha, bm_config_default.zipf_alpha);
    EXPECT_EQ(config_two.min_io_batch_size, bm_config_default.min_io_batch_size);
    EXPECT_EQ(config_two.run_time, bm_config_default.run_time);
    EXPECT_EQ(config_two.latency_sample_frequency, bm_config_default.latency_sample_frequency);
    EXPECT_EQ(config_two.memory_regions[0].percentage_pages_first_partition,
              bm_config_default.memory_regions[0].percentage_pages_first_partition);
  }
}

TEST_F(ConfigTest, ThreadPinningAllNuma) {
  std::vector<SingleBenchmark> single_bms = BenchmarkFactory::create_single_benchmarks(config_thread_multi_core_numa);
  std::vector<ParallelBenchmark> parallel_benchmarks =
      BenchmarkFactory::create_parallel_benchmarks(config_thread_multi_core_numa);
  ASSERT_EQ(single_bms.size(), 1);
  ASSERT_EQ(parallel_benchmarks.size(), 0);

  bm_config = single_bms.at(0).get_benchmark_configs()[0];
  EXPECT_EQ(bm_config.numa_thread_nodes, NumaNodeIDs{42});
  EXPECT_EQ(bm_config.thread_pin_mode, ThreadPinMode::AllNumaCores);
  EXPECT_TRUE(bm_config.thread_core_ids.empty());
}

TEST_F(ConfigTest, ThreadPinningSingleNuma) {
  std::vector<SingleBenchmark> single_bms = BenchmarkFactory::create_single_benchmarks(config_thread_single_core_numa);
  std::vector<ParallelBenchmark> parallel_benchmarks =
      BenchmarkFactory::create_parallel_benchmarks(config_thread_single_core_numa);
  ASSERT_EQ(single_bms.size(), 1);
  ASSERT_EQ(parallel_benchmarks.size(), 0);

  bm_config = single_bms.at(0).get_benchmark_configs()[0];
  EXPECT_EQ(bm_config.numa_thread_nodes, NumaNodeIDs{42});
  EXPECT_EQ(bm_config.thread_pin_mode, ThreadPinMode::SingleNumaCoreIncrement);
  EXPECT_TRUE(bm_config.thread_core_ids.empty());
}

TEST_F(ConfigTest, ThreadPinningSingleFixed) {
  std::vector<SingleBenchmark> single_bms = BenchmarkFactory::create_single_benchmarks(config_thread_single_core);
  std::vector<ParallelBenchmark> parallel_benchmarks =
      BenchmarkFactory::create_parallel_benchmarks(config_thread_single_core);
  ASSERT_EQ(single_bms.size(), 1);
  ASSERT_EQ(parallel_benchmarks.size(), 0);

  bm_config = single_bms.at(0).get_benchmark_configs()[0];
  EXPECT_TRUE(bm_config.numa_thread_nodes.empty());
  EXPECT_EQ(bm_config.thread_pin_mode, ThreadPinMode::SingleCoreFixed);
  EXPECT_EQ(bm_config.thread_core_ids, (CoreIDs{0, 3, 4, 8}));
}

TEST_F(ConfigTest, SingleDecodeInvalidNumaMemoryNodes) {
  // Throw since the `numa_memory_node` YAML field is not a sequence, i.e., [a, b, c], but a single value.
  EXPECT_THROW(BenchmarkFactory::create_single_benchmarks(config_invalid_numa_memory_nodes), mema::MemaException);
}

TEST_F(ConfigTest, SingleDecodeInvalidNumaTaskNodes) {
  // Throw since the `numa_task_node` YAML field is not a sequence, i.e., [ a ], but a single value.
  EXPECT_THROW(BenchmarkFactory::create_single_benchmarks(config_invalid_numa_task_nodes), mema::MemaException);
}

TEST_F(ConfigTest, InvalidSmallAccessSize) {
  bm_config.access_size = 32;
  EXPECT_THROW(bm_config.validate(), MemaException);
  check_log_for_critical("Access Size must be one of");
}

TEST_F(ConfigTest, InvalidPowerAccessSize) {
  bm_config.access_size = 100;
  EXPECT_THROW(bm_config.validate(), MemaException);
  check_log_for_critical("Access Size must be one of");
}

TEST_F(ConfigTest, InvalidMemoryRangeAccessSizeMultiple) {
  bm_config.memory_regions[0].size = 100000;
  EXPECT_THROW(bm_config.validate(), MemaException);
  check_log_for_critical("multiple of access size");
}

TEST_F(ConfigTest, InvalidNumberThreads) {
  bm_config.number_threads = 0;
  EXPECT_THROW(bm_config.validate(), MemaException);
  check_log_for_critical("threads must be");
}

TEST_F(ConfigTest, InvalidMemoryRegionSize) {
  bm_config.access_size = 1024;

  bm_config.memory_regions[0].size = 1 * GiB + 42;
  EXPECT_THROW(bm_config.validate(), MemaException);
  check_log_for_critical("range must be a multiple of access size");
  bm_config.memory_regions[0].size = 1 * GiB;
  EXPECT_NO_THROW(bm_config.validate());

  bm_config.memory_regions[1].size = 1 * GiB + 42;
  EXPECT_THROW(bm_config.validate(), MemaException);
  check_log_for_critical("range must be a multiple of access size");
  bm_config.memory_regions[1].size = 1 * GiB;
  EXPECT_NO_THROW(bm_config.validate());
}

TEST_F(ConfigTest, InvalidMissingMemoryNodes) {
  bm_config.memory_regions[0].size = 1 * GiB;
  bm_config.memory_regions[0].node_ids = {};
  EXPECT_THROW(bm_config.validate(), MemaException);
  check_log_for_critical("memory nodes must be specified");
  bm_config.memory_regions[0].node_ids = {0};
  EXPECT_NO_THROW(bm_config.validate());

  bm_config.memory_regions[1].size = 1 * GiB;
  bm_config.memory_regions[1].node_ids = {};
  EXPECT_THROW(bm_config.validate(), MemaException);
  check_log_for_critical("memory nodes must be specified");
  bm_config.memory_regions[1].node_ids = {0};
  EXPECT_NO_THROW(bm_config.validate());
}

TEST_F(ConfigTest, InvalidPercentageOnFirstNode) {
  bm_config.memory_regions[0].node_ids = {0, 1};

  bm_config.memory_regions[0].percentage_pages_first_partition = 101;
  bm_config.memory_regions[0].node_count_first_partition = 1;
  EXPECT_THROW(bm_config.validate(), MemaException);
  check_log_for_critical("Share of pages located on first node must be in range [0, 100]");
  bm_config.memory_regions[0].percentage_pages_first_partition = 100;
  EXPECT_NO_THROW(bm_config.validate());
  bm_config.memory_regions[0].percentage_pages_first_partition = 0;
  EXPECT_NO_THROW(bm_config.validate());

  // Require two numa nodes when percentage is set.
  bm_config.memory_regions[0].node_ids = {0};
  EXPECT_THROW(bm_config.validate(), MemaException);
  check_log_for_critical(">=2 nodes need to be specified");
}

TEST_F(ConfigTest, InvalidSecondaryMemoryRegion) {
  bm_config.exec_mode = Mode::Custom;
  bm_config.custom_operations = {CustomOp{.memory_type = MemoryType::Secondary, .type = Operation::Read, .size = 64}};
  bm_config.memory_regions[1].size = 0;
  EXPECT_THROW(bm_config.validate(), MemaException);
  check_log_for_critical("secondary_memory_region_size > 0 if the benchmark contains secondary memory operations");
  bm_config.memory_regions[1].size = 1 * GiB;
  EXPECT_NO_THROW(bm_config.validate());
}

TEST_F(ConfigTest, InvalidThreadSingleCoreFixed) {
  bm_config.thread_pin_mode = ThreadPinMode::SingleCoreFixed;
  bm_config.number_threads = 4;
  bm_config.thread_core_ids = {};
  EXPECT_THROW(bm_config.validate(), MemaException);
  check_log_for_critical("Core IDs must be specified if thread pinning is not");
  bm_config.thread_core_ids = {CoreID{1}, CoreID{3}, CoreID{5}};
  EXPECT_THROW(bm_config.validate(), MemaException);
  check_log_for_critical("Number of Core IDs must be greater than or equal thread count.");
  bm_config.number_threads = 3;
  EXPECT_NO_THROW(bm_config.validate());
}

TEST_F(ConfigTest, InvalidNumaBasedThreadPinning) {
  bm_config.thread_pin_mode = ThreadPinMode::AllNumaCores;
  bm_config.numa_thread_nodes = {};
  EXPECT_THROW(bm_config.validate(), MemaException);
  check_log_for_critical("NUMA task nodes must be specified with");
  bm_config.numa_thread_nodes = {NumaNodeID{0}};
  EXPECT_NO_THROW(bm_config.validate());

  bm_config.thread_pin_mode = ThreadPinMode::SingleNumaCoreIncrement;
  bm_config.numa_thread_nodes = {};
  EXPECT_THROW(bm_config.validate(), MemaException);
  check_log_for_critical("NUMA task nodes must be specified with");
  bm_config.numa_thread_nodes = {NumaNodeID{0}};
  EXPECT_NO_THROW(bm_config.validate());
}

TEST_F(ConfigTest, MissingCustomOps) {
  bm_config.exec_mode = Mode::Custom;
  EXPECT_THROW(bm_config.validate(), MemaException);
  check_log_for_critical("Must specify custom_operations");
}

TEST_F(ConfigTest, RandomCustomOps) {
  bm_config.exec_mode = Mode::Random;
  bm_config.custom_operations = {CustomOp{.type = Operation::Read, .size = 64}};
  EXPECT_THROW(bm_config.validate(), MemaException);
  check_log_for_critical("Cannot specify custom_operations");
}

TEST_F(ConfigTest, BadLatencySample) {
  bm_config.exec_mode = Mode::Random;
  bm_config.latency_sample_frequency = 100;
  EXPECT_THROW(bm_config.validate(), MemaException);
  check_log_for_critical("Latency sampling can only");
}

TEST_F(ConfigTest, AsJsonReadSequential) {
  auto bm_config = BenchmarkConfig{};
  bm_config.operation = Operation::Read;
  bm_config.exec_mode = Mode::Sequential;

  const auto json = bm_config.as_json();
  ASSERT_JSON_TRUE(json, contains("m0_region_size"));
  EXPECT_EQ(json["m0_region_size"].get<uint64_t>(), bm_config.memory_regions[0].size);
  ASSERT_JSON_TRUE(json, contains("m0_numa_nodes"));
  EXPECT_EQ(json["m0_numa_nodes"].get<NumaNodeIDs>(), bm_config.memory_regions[0].node_ids);
  ASSERT_JSON_TRUE(json, contains("m0_percentage_pages_first_partition"));
  EXPECT_EQ(json["m0_percentage_pages_first_partition"].get<uint64_t>(), -1);
  ASSERT_JSON_TRUE(json, contains("m0_transparent_huge_pages"));
  EXPECT_EQ(json["m0_transparent_huge_pages"].get<bool>(), bm_config.memory_regions[0].transparent_huge_pages);
  ASSERT_JSON_TRUE(json, contains("m0_explicit_hugepages_size"));
  EXPECT_EQ(json["m0_explicit_hugepages_size"].get<uint64_t>(), bm_config.memory_regions[0].explicit_hugepages_size);
  ASSERT_JSON_TRUE(json, contains("m1_region_size"));
  EXPECT_EQ(json["m1_region_size"].get<uint64_t>(), bm_config.memory_regions[1].size);
  ASSERT_JSON_TRUE(json, contains("m1_numa_nodes"));
  EXPECT_EQ(json["m1_numa_nodes"].get<NumaNodeIDs>(), bm_config.memory_regions[1].node_ids);
  ASSERT_JSON_TRUE(json, contains("m1_percentage_pages_first_partition"));
  EXPECT_EQ(json["m1_percentage_pages_first_partition"].get<uint64_t>(), -1);
  ASSERT_JSON_TRUE(json, contains("m1_transparent_huge_pages"));
  EXPECT_EQ(json["m1_transparent_huge_pages"].get<bool>(), bm_config.memory_regions[1].transparent_huge_pages);
  ASSERT_JSON_TRUE(json, contains("m1_explicit_hugepages_size"));
  EXPECT_EQ(json["m1_explicit_hugepages_size"].get<uint64_t>(), bm_config.memory_regions[1].explicit_hugepages_size);
  ASSERT_JSON_TRUE(json, contains("exec_mode"));
  EXPECT_EQ(json["exec_mode"], "sequential");
  ASSERT_JSON_TRUE(json, contains("numa_task_nodes"));
  EXPECT_EQ(json["numa_task_nodes"].get<NumaNodeIDs>(), bm_config.numa_thread_nodes);
  ASSERT_JSON_TRUE(json, contains("number_threads"));
  EXPECT_EQ(json["number_threads"].get<uint16_t>(), bm_config.number_threads);
  ASSERT_JSON_TRUE(json, contains("min_io_batch_size"));
  EXPECT_EQ(json["min_io_batch_size"].get<uint64_t>(), bm_config.min_io_batch_size);
  // Relevant for non-custom operations
  ASSERT_JSON_TRUE(json, contains("access_size"));
  EXPECT_EQ(json["access_size"].get<uint32_t>(), bm_config.access_size);
  ASSERT_JSON_TRUE(json, contains("operation"));
  EXPECT_EQ(json["operation"], "read");
  // Relevant for writes
  ASSERT_JSON_FALSE(json, contains("cache_instruction"));
  // Relevant for random execution mode
  ASSERT_JSON_FALSE(json, contains("number_operations"));
  ASSERT_JSON_FALSE(json, contains("random_distribution"));
  ASSERT_JSON_FALSE(json, contains("zipf_alpha"));
  // Relevant for custom operations
  ASSERT_JSON_FALSE(json, contains("custom_operations"));
  // Relevant if run_time > 0
  ASSERT_JSON_FALSE(json, contains("run_time"));
}

TEST_F(ConfigTest, AsJsonWriteCustom) {
  auto bm_config = BenchmarkConfig{};
  bm_config.operation = Operation::Write;
  bm_config.exec_mode = Mode::Custom;

  const auto json = bm_config.as_json();
  ASSERT_JSON_TRUE(json, contains("m0_region_size"));
  EXPECT_EQ(json["m0_region_size"].get<uint64_t>(), bm_config.memory_regions[0].size);
  ASSERT_JSON_TRUE(json, contains("m0_numa_nodes"));
  EXPECT_EQ(json["m0_numa_nodes"].get<NumaNodeIDs>(), bm_config.memory_regions[0].node_ids);
  ASSERT_JSON_TRUE(json, contains("m0_percentage_pages_first_partition"));
  EXPECT_EQ(json["m0_percentage_pages_first_partition"].get<uint64_t>(), -1);
  ASSERT_JSON_TRUE(json, contains("m0_transparent_huge_pages"));
  EXPECT_EQ(json["m0_transparent_huge_pages"].get<bool>(), bm_config.memory_regions[0].transparent_huge_pages);
  ASSERT_JSON_TRUE(json, contains("m0_explicit_hugepages_size"));
  EXPECT_EQ(json["m0_explicit_hugepages_size"].get<uint64_t>(), bm_config.memory_regions[0].explicit_hugepages_size);
  ASSERT_JSON_TRUE(json, contains("m1_region_size"));
  EXPECT_EQ(json["m1_region_size"].get<uint64_t>(), bm_config.memory_regions[1].size);
  ASSERT_JSON_TRUE(json, contains("m1_numa_nodes"));
  EXPECT_EQ(json["m1_numa_nodes"].get<NumaNodeIDs>(), bm_config.memory_regions[1].node_ids);
  ASSERT_JSON_TRUE(json, contains("m1_percentage_pages_first_partition"));
  EXPECT_EQ(json["m1_percentage_pages_first_partition"].get<uint64_t>(), -1);
  ASSERT_JSON_TRUE(json, contains("m1_transparent_huge_pages"));
  EXPECT_EQ(json["m1_transparent_huge_pages"].get<bool>(), bm_config.memory_regions[1].transparent_huge_pages);
  ASSERT_JSON_TRUE(json, contains("m1_explicit_hugepages_size"));
  EXPECT_EQ(json["m1_explicit_hugepages_size"].get<uint64_t>(), bm_config.memory_regions[1].explicit_hugepages_size);
  ASSERT_JSON_TRUE(json, contains("exec_mode"));
  EXPECT_EQ(json["exec_mode"], "custom");
  ASSERT_JSON_TRUE(json, contains("numa_task_nodes"));
  EXPECT_EQ(json["numa_task_nodes"].get<NumaNodeIDs>(), bm_config.numa_thread_nodes);
  ASSERT_JSON_TRUE(json, contains("number_threads"));
  EXPECT_EQ(json["number_threads"].get<uint16_t>(), bm_config.number_threads);
  ASSERT_JSON_TRUE(json, contains("min_io_batch_size"));
  EXPECT_EQ(json["min_io_batch_size"].get<uint64_t>(), bm_config.min_io_batch_size);
  // Relevant for non-custom operations
  ASSERT_JSON_FALSE(json, contains("access_size"));
  ASSERT_JSON_FALSE(json, contains("operation"));
  // Relevant for non-custom writes
  ASSERT_JSON_FALSE(json, contains("cache_instruction"));
  // Relevant for random execution mode
  ASSERT_JSON_FALSE(json, contains("random_distribution"));
  ASSERT_JSON_FALSE(json, contains("zipf_alpha"));
  // Relevant for custom operations
  ASSERT_JSON_TRUE(json, contains("number_operations"));
  EXPECT_EQ(json["number_operations"].get<uint64_t>(), bm_config.number_operations);
  ASSERT_JSON_TRUE(json, contains("custom_operations"));
  EXPECT_EQ(json["custom_operations"], (CustomOp::all_to_string(bm_config.custom_operations)));
  // Relevant if run_time > 0
  ASSERT_JSON_FALSE(json, contains("run_time"));
}

TEST_F(ConfigTest, AsJsonWithRuntime) {
  auto bm_config = BenchmarkConfig{};
  bm_config.run_time = 42;
  const auto json = bm_config.as_json();
  ASSERT_JSON_TRUE(json, contains("run_time"));
  EXPECT_EQ(json["run_time"].get<uint64_t>(), 42);
}

}  // namespace mema
