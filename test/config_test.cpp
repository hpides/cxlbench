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
  BenchmarkConfig bm_config;
  static std::filesystem::path test_logger_path;
};

std::filesystem::path ConfigTest::test_logger_path;

TEST_F(ConfigTest, SingleDecodeSequential) {
  std::vector<SingleBenchmark> single_bms = BenchmarkFactory::create_single_benchmarks(config_single_sequential);
  std::vector<ParallelBenchmark> parallel_benchmarks =
      BenchmarkFactory::create_parallel_benchmarks(config_single_sequential);
  ASSERT_EQ(single_bms.size(), 1);
  ASSERT_EQ(parallel_benchmarks.size(), 0);
  bm_config = single_bms.at(0).get_benchmark_configs()[0];

  BenchmarkConfig bm_config_default{};

  EXPECT_EQ(bm_config.memory_region_size, 67108864);
  EXPECT_EQ(bm_config.access_size, 256);
  EXPECT_EQ(bm_config.exec_mode, Mode::Sequential);

  EXPECT_EQ(bm_config.number_threads, 2);

  EXPECT_EQ(bm_config.operation, Operation::Read);
  EXPECT_EQ(bm_config.min_io_chunk_size, 16 * 1024);

  EXPECT_EQ(bm_config.number_operations, bm_config_default.number_operations);
  EXPECT_EQ(bm_config.random_distribution, bm_config_default.random_distribution);
  EXPECT_EQ(bm_config.zipf_alpha, bm_config_default.zipf_alpha);
  EXPECT_EQ(bm_config.flush_instruction, bm_config_default.flush_instruction);
  EXPECT_EQ(bm_config.number_partitions, bm_config_default.number_partitions);
  EXPECT_EQ(bm_config.run_time, bm_config_default.run_time);
  EXPECT_EQ(bm_config.latency_sample_frequency, bm_config_default.latency_sample_frequency);
  EXPECT_EQ(bm_config.numa_memory_nodes, (NumaNodeIDs{0, 1}));
  EXPECT_EQ(bm_config.numa_task_nodes, NumaNodeIDs{0});
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

  EXPECT_EQ(bm_config.memory_region_size, bm_config_default.memory_region_size);
  EXPECT_EQ(bm_config.access_size, bm_config_default.access_size);
  EXPECT_EQ(bm_config.number_operations, bm_config_default.number_operations);
  EXPECT_EQ(bm_config.flush_instruction, bm_config_default.flush_instruction);
  EXPECT_EQ(bm_config.number_partitions, bm_config_default.number_partitions);
  EXPECT_EQ(bm_config.number_threads, bm_config_default.number_threads);
  EXPECT_EQ(bm_config.min_io_chunk_size, bm_config_default.min_io_chunk_size);
  EXPECT_EQ(bm_config.run_time, bm_config_default.run_time);
  EXPECT_EQ(bm_config.latency_sample_frequency, bm_config_default.latency_sample_frequency);
  EXPECT_EQ(bm_config.numa_memory_nodes, NumaNodeIDs{1});
  EXPECT_EQ(bm_config.numa_task_nodes, NumaNodeIDs{0});
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

  EXPECT_EQ(bm_config.memory_region_size, 10737418240);
  EXPECT_EQ(bm_config.access_size, 4096);
  EXPECT_EQ(bm_config.exec_mode, Mode::Random);
  EXPECT_EQ(bm_config.number_operations, 10000000);

  EXPECT_EQ(bm_config.number_threads, 8);

  EXPECT_EQ(bm_config.operation, Operation::Read);

  EXPECT_EQ(bm_config.random_distribution, bm_config_default.random_distribution);
  EXPECT_EQ(bm_config.zipf_alpha, bm_config_default.zipf_alpha);
  EXPECT_EQ(bm_config.flush_instruction, bm_config_default.flush_instruction);
  EXPECT_EQ(bm_config.number_partitions, bm_config_default.number_partitions);
  EXPECT_EQ(bm_config.run_time, bm_config_default.run_time);
  EXPECT_EQ(bm_config.latency_sample_frequency, bm_config_default.latency_sample_frequency);
  EXPECT_EQ(bm_config.numa_memory_nodes, (NumaNodeIDs{0}));
  EXPECT_EQ(bm_config.numa_task_nodes, (NumaNodeIDs{0, 1}));

  bm_config = parallel_bms.at(0).get_benchmark_configs()[1];

  EXPECT_EQ(bm_config.memory_region_size, 10737418240);
  EXPECT_EQ(bm_config.access_size, 256);
  EXPECT_EQ(bm_config.exec_mode, Mode::Sequential);
  EXPECT_EQ(bm_config.flush_instruction, FlushInstruction::None);

  EXPECT_EQ(bm_config.number_threads, 16);

  EXPECT_EQ(bm_config.operation, Operation::Write);

  EXPECT_EQ(bm_config.number_operations, bm_config_default.number_operations);
  EXPECT_EQ(bm_config.random_distribution, bm_config_default.random_distribution);
  EXPECT_EQ(bm_config.zipf_alpha, bm_config_default.zipf_alpha);
  EXPECT_EQ(bm_config.number_partitions, bm_config_default.number_partitions);
  EXPECT_EQ(bm_config.min_io_chunk_size, bm_config_default.min_io_chunk_size);
  EXPECT_EQ(bm_config.run_time, bm_config_default.run_time);
  EXPECT_EQ(bm_config.latency_sample_frequency, bm_config_default.latency_sample_frequency);
  EXPECT_EQ(bm_config.numa_memory_nodes, (NumaNodeIDs{2, 3}));
  EXPECT_EQ(bm_config.numa_task_nodes, (NumaNodeIDs{0, 1}));
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
  for (size_t i = 0; i < bm_count; ++i) {
    const SingleBenchmark& bm = single_bms[i];
    const BenchmarkConfig& config = bm.get_benchmark_configs()[0];

    // Other args are identical for all configs
    EXPECT_EQ(config.memory_region_size, 536870912);
    EXPECT_EQ(config.exec_mode, Mode::Sequential);
    EXPECT_EQ(config.operation, Operation::Read);

    EXPECT_EQ(config.number_operations, bm_config_default.number_operations);
    EXPECT_EQ(config.random_distribution, bm_config_default.random_distribution);
    EXPECT_EQ(config.zipf_alpha, bm_config_default.zipf_alpha);
    EXPECT_EQ(config.flush_instruction, bm_config_default.flush_instruction);
    EXPECT_EQ(config.number_partitions, bm_config_default.number_partitions);
    EXPECT_EQ(config.min_io_chunk_size, bm_config_default.min_io_chunk_size);
    EXPECT_EQ(config.run_time, bm_config_default.run_time);
    EXPECT_EQ(config.latency_sample_frequency, bm_config_default.latency_sample_frequency);
    EXPECT_EQ(config.numa_memory_nodes, (NumaNodeIDs{0, 3}));
    EXPECT_EQ(config.numa_task_nodes, NumaNodeIDs{0});
  }
}

TEST_F(ConfigTest, DecodeCustomOperationsMatrix) {
  const size_t bm_count = 3;
  std::vector<SingleBenchmark> single_bms = BenchmarkFactory::create_single_benchmarks(config_custom_operations_matrix);

  ASSERT_EQ(single_bms.size(), bm_count);
  EXPECT_EQ(single_bms[0].get_benchmark_configs()[0].custom_operations.size(), 3);
  EXPECT_EQ(single_bms[0].get_benchmark_configs()[0].custom_operations[0].type, Operation::Read);
  EXPECT_EQ(single_bms[0].get_benchmark_configs()[0].custom_operations[0].size, 64);
  EXPECT_EQ(single_bms[0].get_benchmark_configs()[0].custom_operations[1].type, Operation::Write);
  EXPECT_EQ(single_bms[0].get_benchmark_configs()[0].custom_operations[1].size, 64);
  EXPECT_EQ(single_bms[0].get_benchmark_configs()[0].custom_operations[1].flush, FlushInstruction::NoCache);
  EXPECT_EQ(single_bms[0].get_benchmark_configs()[0].custom_operations[2].type, Operation::Read);
  EXPECT_EQ(single_bms[0].get_benchmark_configs()[0].custom_operations[2].size, 128);

  EXPECT_EQ(single_bms[1].get_benchmark_configs()[0].custom_operations.size(), 2);
  EXPECT_EQ(single_bms[1].get_benchmark_configs()[0].custom_operations[0].type, Operation::Read);
  EXPECT_EQ(single_bms[1].get_benchmark_configs()[0].custom_operations[0].size, 256);
  EXPECT_EQ(single_bms[1].get_benchmark_configs()[0].custom_operations[1].type, Operation::Write);
  EXPECT_EQ(single_bms[1].get_benchmark_configs()[0].custom_operations[1].size, 256);
  EXPECT_EQ(single_bms[1].get_benchmark_configs()[0].custom_operations[1].flush, FlushInstruction::None);

  EXPECT_EQ(single_bms[2].get_benchmark_configs()[0].custom_operations.size(), 2);
  EXPECT_EQ(single_bms[2].get_benchmark_configs()[0].custom_operations[0].type, Operation::Read);
  EXPECT_EQ(single_bms[2].get_benchmark_configs()[0].custom_operations[0].size, 1024);
  EXPECT_EQ(single_bms[2].get_benchmark_configs()[0].custom_operations[1].type, Operation::Write);
  EXPECT_EQ(single_bms[2].get_benchmark_configs()[0].custom_operations[1].size, 128);
  EXPECT_EQ(single_bms[2].get_benchmark_configs()[0].custom_operations[1].flush, FlushInstruction::Cache);

  BenchmarkConfig bm_config_default{};
  for (size_t i = 0; i < bm_count; ++i) {
    const SingleBenchmark& bm = single_bms[i];
    const BenchmarkConfig& config = bm.get_benchmark_configs()[0];

    // Other args are identical for all configs
    EXPECT_EQ(config.memory_region_size, 2147483648);
    EXPECT_EQ(config.exec_mode, Mode::Custom);
    EXPECT_EQ(config.number_operations, 100000000);
    EXPECT_EQ(config.number_threads, 16);

    EXPECT_EQ(config.random_distribution, bm_config_default.random_distribution);
    EXPECT_EQ(config.zipf_alpha, bm_config_default.zipf_alpha);
    EXPECT_EQ(config.number_partitions, bm_config_default.number_partitions);
    EXPECT_EQ(config.min_io_chunk_size, bm_config_default.min_io_chunk_size);
    EXPECT_EQ(config.run_time, bm_config_default.run_time);
    EXPECT_EQ(config.latency_sample_frequency, bm_config_default.latency_sample_frequency);
    EXPECT_TRUE(bm_config.numa_memory_nodes.empty());
    EXPECT_TRUE(bm_config.numa_task_nodes.empty());
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
  for (size_t i = 0; i < bm_count; ++i) {
    const ParallelBenchmark& bm = parallel_bms[i];
    const BenchmarkConfig& config_one = bm.get_benchmark_configs()[0];
    const BenchmarkConfig& config_two = bm.get_benchmark_configs()[1];

    // Other args are identical for all configs
    EXPECT_EQ(config_one.memory_region_size, 10737418240);
    EXPECT_EQ(config_one.access_size, 4096);
    EXPECT_EQ(config_one.exec_mode, Mode::Random);
    EXPECT_EQ(config_one.number_operations, 10000000);
    EXPECT_EQ(config_one.operation, Operation::Read);

    EXPECT_EQ(config_one.random_distribution, bm_config_default.random_distribution);
    EXPECT_EQ(config_one.zipf_alpha, bm_config_default.zipf_alpha);
    EXPECT_EQ(config_one.flush_instruction, bm_config_default.flush_instruction);
    EXPECT_EQ(config_one.number_partitions, bm_config_default.number_partitions);
    EXPECT_EQ(config_one.min_io_chunk_size, bm_config_default.min_io_chunk_size);
    EXPECT_EQ(config_one.run_time, bm_config_default.run_time);
    EXPECT_EQ(config_one.latency_sample_frequency, bm_config_default.latency_sample_frequency);
    EXPECT_TRUE(bm_config.numa_memory_nodes.empty());
    EXPECT_TRUE(bm_config.numa_task_nodes.empty());

    EXPECT_EQ(config_two.memory_region_size, 10737418240);
    EXPECT_EQ(config_two.exec_mode, Mode::Sequential);
    EXPECT_EQ(config_two.operation, Operation::Write);
    EXPECT_EQ(config_two.number_threads, 16);
    EXPECT_EQ(config_two.flush_instruction, FlushInstruction::None);

    EXPECT_EQ(config_two.number_operations, bm_config_default.number_operations);
    EXPECT_EQ(config_two.random_distribution, bm_config_default.random_distribution);
    EXPECT_EQ(config_two.zipf_alpha, bm_config_default.zipf_alpha);
    EXPECT_EQ(config_two.number_partitions, bm_config_default.number_partitions);
    EXPECT_EQ(config_two.min_io_chunk_size, bm_config_default.min_io_chunk_size);
    EXPECT_EQ(config_two.run_time, bm_config_default.run_time);
    EXPECT_EQ(config_two.latency_sample_frequency, bm_config_default.latency_sample_frequency);
    EXPECT_TRUE(bm_config.numa_memory_nodes.empty());
    EXPECT_TRUE(bm_config.numa_task_nodes.empty());
  }
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
  check_log_for_critical("at least 64-byte");
}

TEST_F(ConfigTest, InvalidPowerAccessSize) {
  bm_config.access_size = 100;
  EXPECT_THROW(bm_config.validate(), MemaException);
  check_log_for_critical("power of 2");
}

TEST_F(ConfigTest, InvalidMemoryRangeAccessSizeMultiple) {
  bm_config.memory_region_size = 100000;
  EXPECT_THROW(bm_config.validate(), MemaException);
  check_log_for_critical("multiple of access size");
}

TEST_F(ConfigTest, InvalidNumberThreads) {
  bm_config.number_threads = 0;
  EXPECT_THROW(bm_config.validate(), MemaException);
  check_log_for_critical("threads must be");
}

TEST_F(ConfigTest, InvalidSmallThreadPartitionRatio) {
  bm_config.number_partitions = 5;
  bm_config.number_threads = 12;
  EXPECT_THROW(bm_config.validate(), MemaException);
  check_log_for_critical("threads must be a multiple of number partitions");
}

TEST_F(ConfigTest, InvalidLargeThreadPartitionRatio) {
  bm_config.number_partitions = 2;
  bm_config.number_threads = 1;
  EXPECT_THROW(bm_config.validate(), MemaException);
  check_log_for_critical("threads must be a multiple of number partitions");
}

TEST_F(ConfigTest, BadNumberPartitionSplit) {
  bm_config.number_threads = 36;
  bm_config.number_partitions = 36;
  EXPECT_THROW(bm_config.validate(), MemaException);
  check_log_for_critical("evenly divisible into");
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
  ASSERT_JSON_TRUE(json, contains("memory_region_size"));
  EXPECT_EQ(json["memory_region_size"].get<uint64_t>(), bm_config.memory_region_size);
  ASSERT_JSON_TRUE(json, contains("exec_mode"));
  EXPECT_EQ(json["exec_mode"], "sequential");
  ASSERT_JSON_TRUE(json, contains("numa_memory_nodes"));
  EXPECT_EQ(json["numa_memory_nodes"].get<NumaNodeIDs>(), bm_config.numa_memory_nodes);
  ASSERT_JSON_TRUE(json, contains("numa_task_nodes"));
  EXPECT_EQ(json["numa_task_nodes"].get<NumaNodeIDs>(), bm_config.numa_task_nodes);
  ASSERT_JSON_TRUE(json, contains("number_partitions"));
  EXPECT_EQ(json["number_partitions"].get<uint16_t>(), bm_config.number_partitions);
  ASSERT_JSON_TRUE(json, contains("number_threads"));
  EXPECT_EQ(json["number_threads"].get<uint16_t>(), bm_config.number_threads);
  ASSERT_JSON_TRUE(json, contains("min_io_chunk_size"));
  EXPECT_EQ(json["min_io_chunk_size"].get<uint64_t>(), bm_config.min_io_chunk_size);
  ASSERT_JSON_TRUE(json, contains("transparent_huge_pages"));
  EXPECT_EQ(json["transparent_huge_pages"].get<bool>(), bm_config.transparent_huge_pages);
  ASSERT_JSON_TRUE(json, contains("explicit_hugepages_size"));
  EXPECT_EQ(json["explicit_hugepages_size"].get<uint64_t>(), bm_config.explicit_hugepages_size);
  // Relevant for non-custom operations
  ASSERT_JSON_TRUE(json, contains("access_size"));
  EXPECT_EQ(json["access_size"].get<uint32_t>(), bm_config.access_size);
  ASSERT_JSON_TRUE(json, contains("operation"));
  EXPECT_EQ(json["operation"], "read");
  // Relevant for writes
  ASSERT_JSON_FALSE(json, contains("flush_instruction"));
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
  ASSERT_JSON_TRUE(json, contains("memory_region_size"));
  EXPECT_EQ(json["memory_region_size"].get<uint64_t>(), bm_config.memory_region_size);
  ASSERT_JSON_TRUE(json, contains("exec_mode"));
  EXPECT_EQ(json["exec_mode"], "custom");
  ASSERT_JSON_TRUE(json, contains("numa_memory_nodes"));
  EXPECT_EQ(json["numa_memory_nodes"].get<NumaNodeIDs>(), bm_config.numa_memory_nodes);
  ASSERT_JSON_TRUE(json, contains("numa_task_nodes"));
  EXPECT_EQ(json["numa_task_nodes"].get<NumaNodeIDs>(), bm_config.numa_task_nodes);
  ASSERT_JSON_TRUE(json, contains("number_partitions"));
  EXPECT_EQ(json["number_partitions"].get<uint16_t>(), bm_config.number_partitions);
  ASSERT_JSON_TRUE(json, contains("number_threads"));
  EXPECT_EQ(json["number_threads"].get<uint16_t>(), bm_config.number_threads);
  ASSERT_JSON_TRUE(json, contains("min_io_chunk_size"));
  EXPECT_EQ(json["min_io_chunk_size"].get<uint64_t>(), bm_config.min_io_chunk_size);
  ASSERT_JSON_TRUE(json, contains("transparent_huge_pages"));
  EXPECT_EQ(json["transparent_huge_pages"].get<bool>(), bm_config.transparent_huge_pages);
  ASSERT_JSON_TRUE(json, contains("explicit_hugepages_size"));
  EXPECT_EQ(json["explicit_hugepages_size"].get<uint64_t>(), bm_config.explicit_hugepages_size);
  // Relevant for non-custom operations
  ASSERT_JSON_FALSE(json, contains("access_size"));
  ASSERT_JSON_FALSE(json, contains("operation"));
  // Relevant for non-custom writes
  ASSERT_JSON_FALSE(json, contains("flush_instruction"));
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

TEST_F(ConfigTest, ToString) {
  auto bm_config = BenchmarkConfig{};

  // bm_config.
  bm_config.memory_region_size = 40 * GIBIBYTES_IN_BYTES;
  bm_config.exec_mode = Mode::Random;
  bm_config.numa_memory_nodes = {1, 2};
  bm_config.numa_task_nodes = {3, 4};
  bm_config.number_partitions = 8;
  bm_config.number_threads = 4;
  bm_config.min_io_chunk_size = 32 * MEBIBYTES_IN_BYTES;
  bm_config.access_size = 512;
  bm_config.operation = Operation::Write;
  bm_config.flush_instruction = FlushInstruction::Cache;
  bm_config.number_operations = 50'000'000;
  bm_config.random_distribution = RandomDistribution::Zipf;
  bm_config.zipf_alpha = 0.8;

  std::string expected_output =
      "memory range: 42949672960, exec mode: random, "
      "memory numa nodes: [1, 2], task numa nodes: [3, 4], partition count: 8, thread count: 4, min io chunk size: "
      "33554432, access size: 512, "
      "operation: write, flush instruction: cache, number operations: 50000000, random distribution: zipf, "
      "zipf alpha: 0.8";
  EXPECT_EQ(bm_config.to_string(), expected_output);
}

}  // namespace mema
