#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <initializer_list>
#include <iostream>
#include <mutex>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>

#include "benchmark/benchmark.h"
#include "common.hpp"
#include "simd.hpp"

using RowId = uint32_t;
using IntEntry = uint32_t;

using IntColumn = AlignedData<IntEntry, 64>;
using MatchingRows = AlignedData<RowId, 64>;

static constexpr size_t COLUMN_SIZE_KiB = 1024ul * 512;
// static constexpr size_t COLUMN_SIZE_KiB = 1024ul * 1024ul * 2;
static constexpr size_t NUM_BASE_ROWS = 16;  // 16 * 4 Byte values = 64 Byte (AVX512 register size)
static constexpr size_t SCALE_FACTOR_1KiB = 1024 / (NUM_BASE_ROWS * sizeof(IntEntry));
static constexpr size_t NUM_ROWS = COLUMN_SIZE_KiB * SCALE_FACTOR_1KiB * NUM_BASE_ROWS;
static constexpr size_t NUM_UNIQUE_VALUES = 1024;

using NodeID = uint32_t;
constexpr NodeID INVALID_NODE = std::numeric_limits<NodeID>::max();

struct Placement {
  std::vector<NodeID> thread_nodes;
  std::vector<NodeID> local_nodes;
  std::vector<NodeID> remote_nodes;
  float share_remote_columns;
  bool matching_rows_remote;
};

// Profiling
void execute_command(const std::string& command) {
  int ret_code = std::system(command.c_str());
  if (ret_code != 0) {
    std::cerr << "Command failed: " << command << "\n";
  }
}

std::string vtune_dir = "";

static void DoSetup(const benchmark::State& /*unused*/) {
  const char* vtune_dir_cstr = std::getenv("VTUNE_DIR");
  vtune_dir = vtune_dir_cstr ? std::string(vtune_dir_cstr) : "";
  if (!vtune_dir.empty()) {
    execute_command("vtune -command resume -r " + vtune_dir);
    std::cout << "vtune dir: " << vtune_dir << std::endl;
  }
}

static void DoTeardown(const benchmark::State& /*unused*/) {
  execute_command("vtune -command pause -r " + vtune_dir);
  vtune_dir = "";
}

/*
 * Builds a lookup table that, given a comparison-result bitmask, returns the indices of the matching elements
 * compressed to the front. Can be used as a shuffle mask for source-selecting shuffles. Examples:
 * [0 0 0 1] -> [0, unused_index, unused_index, unused_index]
 * [1 0 1 0] -> [1, 3, unused_index, unused_index]
 */
template <size_t ComparisonResultBits, typename IndexT, IndexT unused_index>
static constexpr auto lookup_table_for_compressed_offsets_by_comparison_result() {
  std::array<std::array<IndexT, ComparisonResultBits>, 1 << ComparisonResultBits> lookup_table{};

  for (size_t index = 0; index < lookup_table.size(); ++index) {
    auto& shuffle_mask = lookup_table[index];
    std::fill(shuffle_mask.begin(), shuffle_mask.end(), unused_index);

    size_t first_empty_output_slot = 0;
    for (size_t comparison_result_rest = index; comparison_result_rest != 0;
         comparison_result_rest &= comparison_result_rest - 1) {
      shuffle_mask[first_empty_output_slot++] = std::countr_zero(comparison_result_rest);
    }
  }

  return lookup_table;
}

/*
 * SSE and NEON do not allow shuffling elements with run-time masks, so we have to create byte shuffle masks.
 * This transforms a shuffle mask like [1, 3, unused_index, unused_index] for `uint32_t`s to the byte mask
 * [4, 5, 6, 7,   12, 13, 14, 15,   U, U, U, U,   U, U, U, U ]
 */
template <typename VectorElementT, typename IndexT, IndexT unused_index>
static constexpr auto element_shuffle_table_to_byte_shuffle_table(auto element_shuffle_table) {
  static_assert(std::endian::native == std::endian::little, "Probably doesn't work for big-endian systems.");
  constexpr size_t OUTPUT_ELEMENTS_PER_MASK = sizeof(VectorElementT) * element_shuffle_table[0].size();
  std::array<std::array<IndexT, OUTPUT_ELEMENTS_PER_MASK>, element_shuffle_table.size()> byte_shuffle_table{};

  for (size_t row = 0; row < element_shuffle_table.size(); ++row) {
    const auto& element_shuffle_mask = element_shuffle_table[row];
    auto& byte_shuffle_mask = byte_shuffle_table[row];

    for (size_t element_index = 0; element_index < element_shuffle_mask.size(); ++element_index) {
      const auto element_mask_value = element_shuffle_mask[element_index];
      IndexT* byte_mask_group_begin = byte_shuffle_mask.data() + element_index * sizeof(VectorElementT);
      IndexT* byte_mask_group_end = byte_mask_group_begin + sizeof(VectorElementT);

      if (element_mask_value == unused_index) {
        std::fill(byte_mask_group_begin, byte_mask_group_end, unused_index);
      } else {
        std::iota(byte_mask_group_begin, byte_mask_group_end, element_mask_value * sizeof(VectorElementT));
      }
    }
  }

  return byte_shuffle_table;
}

struct naive_scan {
  RowId operator()(const IntColumn& column, IntEntry filter_val, MatchingRows* matching_rows) {
    const IntEntry* column_data = column.aligned_data();
    RowId* output = matching_rows->aligned_data();

    RowId num_matching_rows = 0;
    for (RowId row = 0; row < NUM_ROWS; ++row) {
      if (column_data[row] < filter_val) {
        output[num_matching_rows++] = row;
      }
    }
    return num_matching_rows;
  }
};

#if AVX512_AVAILABLE
struct x86_avx512_512_scan {
  static constexpr uint32_t NUM_MATCHES_PER_VECTOR = sizeof(__m512i) / sizeof(IntEntry);

  RowId operator()(const IntColumn& column, IntEntry filter_val, MatchingRows* matching_rows) {
    const IntEntry* __restrict rows = column.aligned_data();
    RowId* __restrict output = matching_rows->aligned_data();

    const __m512i filter_vec = _mm512_set1_epi32(static_cast<int>(filter_val));
    const __m512i row_id_offsets = _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);

    RowId num_matching_rows = 0;
    static_assert(NUM_ROWS % NUM_MATCHES_PER_VECTOR == 0);
    for (RowId chunk_start_row = 0; chunk_start_row < NUM_ROWS; chunk_start_row += NUM_MATCHES_PER_VECTOR) {
      // x86: Doing this instead of {start_row + 0, start_row + 1, ...} has a 3x performance improvement! Also applies
      // to the gcc-vec versions.
      const __m512i row_ids = _mm512_add_epi32(_mm512_set1_epi32(static_cast<int>(chunk_start_row)), row_id_offsets);

      const __m512i rows_to_match = _mm512_load_epi32(rows + chunk_start_row);
      const __mmask16 matches = _mm512_cmplt_epi32_mask(rows_to_match, filter_vec);

      // X86512ScanStrategy::COMPRESSSTORE
      _mm512_mask_compressstoreu_epi32(output + num_matching_rows, matches, row_ids);
      // X86512ScanStrategy::COMPRESS_PLUS_STORE
      // auto compressed_rows = _mm512_mask_compress_epi32(row_ids, matches, row_ids);
      // _mm512_storeu_epi32(output + num_matching_rows, compressed_rows);

      num_matching_rows += std::popcount(matches);
    }

    return num_matching_rows;
  }
};
#endif

template <typename ScanFn>
void BM_int_scan(benchmark::State& state, const Placement& placement) {
  const auto num_columns_remote = static_cast<int32_t>(placement.share_remote_columns * state.threads());
  // printf("thread idx: %u\n", state.thread_index());
  // printf("thread count: %u\n", state.threads());
  // printf("remote share: %f\n", placement.share_remote_columns);
  // printf("columns remote: %u\n", num_columns_remote);
  pin_thread(placement.thread_nodes);
  const auto& column_nodes = state.thread_index() < num_columns_remote ? placement.remote_nodes : placement.local_nodes;
  // std::stringstream ss;
  // ss << " column nodes: ";
  // for (auto& a : column_nodes) {
  //   ss << a;
  // }
  // printf("T %u, %s\n", state.thread_index(), ss.str().c_str());
  IntColumn column{NUM_ROWS, column_nodes};
  const auto& matching_rows_nodes = placement.matching_rows_remote ? placement.remote_nodes : placement.local_nodes;
  MatchingRows matching_rows{NUM_ROWS, matching_rows_nodes};

  static_assert(NUM_ROWS % NUM_UNIQUE_VALUES == 0, "Number of rows must be a multiple of num unique values.");
  const int64_t input_per_thousand = state.range(0);
  const auto percentage_to_pass_filter = static_cast<double>(input_per_thousand) / 1000;

  // Our filter value comparison is `row < filter_value`, so we can control the selectivity as follows:
  //   For percentage =   0, the filter value is                     0, i.e., no values will match.
  //   For percentage =  50, the filter value is NUM_UNIQUE_VALUES / 2, i.e., 50% of all values will match.
  //   For percentage = 100, the filter value is     NUM_UNIQUE_VALUES, i.e., all values will match.
  const auto filter_value = static_cast<IntEntry>(NUM_UNIQUE_VALUES * percentage_to_pass_filter);

  IntEntry* column_data = column.aligned_data();
  for (size_t i = 0; i < NUM_ROWS; ++i) {
    column_data[i] = i % NUM_UNIQUE_VALUES;
  }
  std::mt19937 rng{std::random_device{}()};
  std::shuffle(column_data, column_data + NUM_ROWS, rng);

  // Correctness check with naive implementation
  ScanFn scan_fn{};
  MatchingRows matching_rows_naive{NUM_ROWS, {0}};
  const RowId num_matches_naive = naive_scan{}(column, filter_value, &matching_rows_naive);
  const RowId num_matches_specialized = scan_fn(column, filter_value, &matching_rows);

  if (num_matches_naive != num_matches_specialized) {
    throw std::runtime_error{"Bad result. Expected " + std::to_string(num_matches_naive) + " rows to match, but got " +
                             std::to_string(num_matches_specialized)};
  }
  for (size_t i = 0; i < num_matches_naive; ++i) {
    if (matching_rows_naive.aligned_data()[i] != matching_rows.aligned_data()[i]) {
      throw std::runtime_error{"Bad result compare at position: " + std::to_string(i)};
    }
  }

  // Sanity check that the 1000 and 0 per_thousand math works out.
  if (input_per_thousand == 1000 && num_matches_specialized != NUM_ROWS) {
    throw std::runtime_error{"Bad result. Did not match all rows."};
  }
  if (input_per_thousand == 0 && num_matches_specialized != 0) {
    throw std::runtime_error{"Bad result. Did not match 0 rows."};
  }

  benchmark::DoNotOptimize(column.aligned_data());
  benchmark::DoNotOptimize(matching_rows.aligned_data());
  clear_caches();

  for (auto _ : state) {
    RowId num_matches = scan_fn(column, filter_value, &matching_rows);
    benchmark::DoNotOptimize(num_matches);
  }

  // printf("T%u\n", state.thread_index());
  // state.SetItemsProcessed(state.items_processed() + NUM_ROWS);
  state.counters["scanned_values"] = NUM_ROWS;
  // state.counters["time_per_value"] = benchmark::Counter(static_cast<double>(state.iterations() * NUM_ROWS),
  //                                                       benchmark::Counter::kIsRate | benchmark::Counter::kInvert);
  // state.counters["values_per_second"] =
  //     benchmark::Counter(static_cast<double>(state.iterations() * NUM_ROWS), benchmark::Counter::kIsRate);

  // state.counters["thp_T" + std::to_string(state.thread_index())] =
  //     benchmark::Counter(static_cast<double>(state.iterations() * NUM_ROWS), benchmark::Counter::kIsRate);
}

// #define BM_ARGS
// Unit(benchmark::kMicrosecond)->Arg(0)->Arg(10)->Arg(33)->Arg(50)->Arg(66)->Arg(100)->ReportAggregatesOnly()
// #define BM_ARGS
// Unit(benchmark::kMicrosecond)->Repetitions(1)->Iterations(1)->ThreadRange(1,16)->Arg(1000)->Arg(500)->Arg(100)->Arg(10)->Arg(1)->Arg(0)->ReportAggregatesOnly()

#define ARGS_E1                 \
  Unit(benchmark::kMicrosecond) \
      ->Iterations(1)           \
      ->Threads(1)              \
      ->Threads(2)              \
      ->Threads(4)              \
      ->Threads(8)              \
      ->Threads(16)             \
      ->Threads(24)             \
      ->Threads(32)             \
      ->Threads(40)             \
      ->Threads(48)             \
      ->UseRealTime()           \
      ->Arg(1000)               \
      ->Arg(600)                \
      ->Arg(200)                \
      ->Arg(1)

// BENCHMARK(BM_int_scan<naive_scan>)->BM_ARGS;

// struct Placement {
//   std::vector<NodeID> thread_nodes;
//   std::vector<NodeID> local_nodes;
//   std::vector<NodeID> remote_nodes;
//   float share_remote_columns;
//   bool matching_rows_remote;
// };

// TODO(user) Configure this according to your configuration of interest.
// #define ARGS_E2 Unit(benchmark::kMicrosecond)->Iterations(1)->Threads(40)->UseRealTime()
#define ARGS_E2 Unit(benchmark::kMicrosecond)->Iterations(1)->Threads(40)->Threads(20)->Threads(10)->UseRealTime()

#if AVX512_AVAILABLE
// Experiment E1
BENCHMARK_TEMPLATE1_CAPTURE(BM_int_scan, x86_avx512_512_scan, E1_AllLocal, {{1}, {1}, {INVALID_NODE}, 0.0f, false})
    ->ARGS_E1;
BENCHMARK_TEMPLATE1_CAPTURE(BM_int_scan, x86_avx512_512_scan, E1_ColumnsCXL1Blade, {{1}, {1}, {2}, 1.0f, false})
    ->ARGS_E1;
BENCHMARK_TEMPLATE1_CAPTURE(BM_int_scan, x86_avx512_512_scan, E1_ColumnsCXL4Blades,
                            {{1}, {1}, {2, 3, 4, 5}, 1.0f, false})
    ->ARGS_E1;
BENCHMARK_TEMPLATE1_CAPTURE(BM_int_scan, x86_avx512_512_scan, E1_AllCXL1Blade, {{1}, {1}, {2}, 1.0f, true})->ARGS_E1;
BENCHMARK_TEMPLATE1_CAPTURE(BM_int_scan, x86_avx512_512_scan, E1_AllCXL4Blades, {{1}, {1}, {2, 3, 4, 5}, 1.0f, true})
    ->ARGS_E1;

// E1 Performance Analysis (E1PA)
#define ARGS_E1PA \
  Unit(benchmark::kMicrosecond)->Iterations(1)->Threads(40)->UseRealTime()->Setup(DoSetup)->Teardown(DoTeardown)
// Selectivity 100%
BENCHMARK_TEMPLATE1_CAPTURE(BM_int_scan, x86_avx512_512_scan, E1PA_AllLocalT40S1000,
                            {{1}, {1}, {INVALID_NODE}, 0.0f, false})
    ->ARGS_E1PA->Arg(1000);
BENCHMARK_TEMPLATE1_CAPTURE(BM_int_scan, x86_avx512_512_scan, E1PA_AllCXL1BladeT40S1000, {{1}, {1}, {2}, 1.0f, true})
    ->ARGS_E1PA->Arg(1000);
BENCHMARK_TEMPLATE1_CAPTURE(BM_int_scan, x86_avx512_512_scan, E1PA_AllCXL4BladesT40S1000,
                            {{1}, {1}, {2, 3, 4, 5}, 1.0f, true})
    ->ARGS_E1PA->Arg(1000);
// Selectivity 0.1%
BENCHMARK_TEMPLATE1_CAPTURE(BM_int_scan, x86_avx512_512_scan, E1PA_AllLocalT40S1,
                            {{1}, {1}, {INVALID_NODE}, 0.0f, false})
    ->ARGS_E1PA->Arg(1);
BENCHMARK_TEMPLATE1_CAPTURE(BM_int_scan, x86_avx512_512_scan, E1PA_AllCXL1BladeT40S1, {{1}, {1}, {2}, 1.0f, true})
    ->ARGS_E1PA->Arg(1);
BENCHMARK_TEMPLATE1_CAPTURE(BM_int_scan, x86_avx512_512_scan, E1PA_AllCXL4BladesT40S1,
                            {{1}, {1}, {2, 3, 4, 5}, 1.0f, true})
    ->ARGS_E1PA->Arg(1);

// Experiment E2 - share of columns on CXL, TIDs on local memory
BENCHMARK_TEMPLATE1_CAPTURE(BM_int_scan, x86_avx512_512_scan, E2_ColumnsCXL0_1Blade, {{1}, {1}, {2}, .0f, false})
    ->ARGS_E2->Arg(1000);
BENCHMARK_TEMPLATE1_CAPTURE(BM_int_scan, x86_avx512_512_scan, E2_ColumnsCXL10_1Blade, {{1}, {1}, {2}, .1f, false})
    ->ARGS_E2->Arg(1000);
BENCHMARK_TEMPLATE1_CAPTURE(BM_int_scan, x86_avx512_512_scan, E2_ColumnsCXL20_1Blade, {{1}, {1}, {2}, .2f, false})
    ->ARGS_E2->Arg(1000);
BENCHMARK_TEMPLATE1_CAPTURE(BM_int_scan, x86_avx512_512_scan, E2_ColumnsCXL30_1Blade, {{1}, {1}, {2}, .3f, false})
    ->ARGS_E2->Arg(1000);
BENCHMARK_TEMPLATE1_CAPTURE(BM_int_scan, x86_avx512_512_scan, E2_ColumnsCXL40_1Blade, {{1}, {1}, {2}, .4f, false})
    ->ARGS_E2->Arg(1000);
BENCHMARK_TEMPLATE1_CAPTURE(BM_int_scan, x86_avx512_512_scan, E2_ColumnsCXL50_1Blade, {{1}, {1}, {2}, .5f, false})
    ->ARGS_E2->Arg(1000);
BENCHMARK_TEMPLATE1_CAPTURE(BM_int_scan, x86_avx512_512_scan, E2_ColumnsCXL60_1Blade, {{1}, {1}, {2}, .6f, false})
    ->ARGS_E2->Arg(1000);
BENCHMARK_TEMPLATE1_CAPTURE(BM_int_scan, x86_avx512_512_scan, E2_ColumnsCXL70_1Blade, {{1}, {1}, {2}, .7f, false})
    ->ARGS_E2->Arg(1000);
BENCHMARK_TEMPLATE1_CAPTURE(BM_int_scan, x86_avx512_512_scan, E2_ColumnsCXL80_1Blade, {{1}, {1}, {2}, .8f, false})
    ->ARGS_E2->Arg(1000);
BENCHMARK_TEMPLATE1_CAPTURE(BM_int_scan, x86_avx512_512_scan, E2_ColumnsCXL90_1Blade, {{1}, {1}, {2}, .9f, false})
    ->ARGS_E2->Arg(1000);
BENCHMARK_TEMPLATE1_CAPTURE(BM_int_scan, x86_avx512_512_scan, E2_ColumnsCXL100_1Blade, {{1}, {1}, {2}, 1.0f, false})
    ->ARGS_E2->Arg(1000);

BENCHMARK_TEMPLATE1_CAPTURE(BM_int_scan, x86_avx512_512_scan, E2_ColumnsCXL0_1Blade, {{1}, {1}, {2}, .0f, false})
    ->ARGS_E2->Arg(1);
BENCHMARK_TEMPLATE1_CAPTURE(BM_int_scan, x86_avx512_512_scan, E2_ColumnsCXL10_1Blade, {{1}, {1}, {2}, .1f, false})
    ->ARGS_E2->Arg(1);
BENCHMARK_TEMPLATE1_CAPTURE(BM_int_scan, x86_avx512_512_scan, E2_ColumnsCXL20_1Blade, {{1}, {1}, {2}, .2f, false})
    ->ARGS_E2->Arg(1);
BENCHMARK_TEMPLATE1_CAPTURE(BM_int_scan, x86_avx512_512_scan, E2_ColumnsCXL30_1Blade, {{1}, {1}, {2}, .3f, false})
    ->ARGS_E2->Arg(1);
BENCHMARK_TEMPLATE1_CAPTURE(BM_int_scan, x86_avx512_512_scan, E2_ColumnsCXL40_1Blade, {{1}, {1}, {2}, .4f, false})
    ->ARGS_E2->Arg(1);
BENCHMARK_TEMPLATE1_CAPTURE(BM_int_scan, x86_avx512_512_scan, E2_ColumnsCXL50_1Blade, {{1}, {1}, {2}, .5f, false})
    ->ARGS_E2->Arg(1);
BENCHMARK_TEMPLATE1_CAPTURE(BM_int_scan, x86_avx512_512_scan, E2_ColumnsCXL60_1Blade, {{1}, {1}, {2}, .6f, false})
    ->ARGS_E2->Arg(1);
BENCHMARK_TEMPLATE1_CAPTURE(BM_int_scan, x86_avx512_512_scan, E2_ColumnsCXL70_1Blade, {{1}, {1}, {2}, .7f, false})
    ->ARGS_E2->Arg(1);
BENCHMARK_TEMPLATE1_CAPTURE(BM_int_scan, x86_avx512_512_scan, E2_ColumnsCXL80_1Blade, {{1}, {1}, {2}, .8f, false})
    ->ARGS_E2->Arg(1);
BENCHMARK_TEMPLATE1_CAPTURE(BM_int_scan, x86_avx512_512_scan, E2_ColumnsCXL90_1Blade, {{1}, {1}, {2}, .9f, false})
    ->ARGS_E2->Arg(1);
BENCHMARK_TEMPLATE1_CAPTURE(BM_int_scan, x86_avx512_512_scan, E2_ColumnsCXL100_1Blade, {{1}, {1}, {2}, 1.0f, false})
    ->ARGS_E2->Arg(1);

BENCHMARK_TEMPLATE1_CAPTURE(BM_int_scan, x86_avx512_512_scan, E2_ColumnsCXL0_4Blade,
                            {{1}, {1}, {2, 3, 4, 5}, .0f, false})
    ->ARGS_E2->Arg(1000);
BENCHMARK_TEMPLATE1_CAPTURE(BM_int_scan, x86_avx512_512_scan, E2_ColumnsCXL10_4Blade,
                            {{1}, {1}, {2, 3, 4, 5}, .1f, false})
    ->ARGS_E2->Arg(1000);
BENCHMARK_TEMPLATE1_CAPTURE(BM_int_scan, x86_avx512_512_scan, E2_ColumnsCXL20_4Blade,
                            {{1}, {1}, {2, 3, 4, 5}, .2f, false})
    ->ARGS_E2->Arg(1000);
BENCHMARK_TEMPLATE1_CAPTURE(BM_int_scan, x86_avx512_512_scan, E2_ColumnsCXL30_4Blade,
                            {{1}, {1}, {2, 3, 4, 5}, .3f, false})
    ->ARGS_E2->Arg(1000);
BENCHMARK_TEMPLATE1_CAPTURE(BM_int_scan, x86_avx512_512_scan, E2_ColumnsCXL40_4Blade,
                            {{1}, {1}, {2, 3, 4, 5}, .4f, false})
    ->ARGS_E2->Arg(1000);
BENCHMARK_TEMPLATE1_CAPTURE(BM_int_scan, x86_avx512_512_scan, E2_ColumnsCXL50_4Blade,
                            {{1}, {1}, {2, 3, 4, 5}, .5f, false})
    ->ARGS_E2->Arg(1000);
BENCHMARK_TEMPLATE1_CAPTURE(BM_int_scan, x86_avx512_512_scan, E2_ColumnsCXL60_4Blade,
                            {{1}, {1}, {2, 3, 4, 5}, .6f, false})
    ->ARGS_E2->Arg(1000);
BENCHMARK_TEMPLATE1_CAPTURE(BM_int_scan, x86_avx512_512_scan, E2_ColumnsCXL70_4Blade,
                            {{1}, {1}, {2, 3, 4, 5}, .7f, false})
    ->ARGS_E2->Arg(1000);
BENCHMARK_TEMPLATE1_CAPTURE(BM_int_scan, x86_avx512_512_scan, E2_ColumnsCXL80_4Blade,
                            {{1}, {1}, {2, 3, 4, 5}, .8f, false})
    ->ARGS_E2->Arg(1000);
BENCHMARK_TEMPLATE1_CAPTURE(BM_int_scan, x86_avx512_512_scan, E2_ColumnsCXL90_4Blade,
                            {{1}, {1}, {2, 3, 4, 5}, .9f, false})
    ->ARGS_E2->Arg(1000);
BENCHMARK_TEMPLATE1_CAPTURE(BM_int_scan, x86_avx512_512_scan, E2_ColumnsCXL100_4Blade,
                            {{1}, {1}, {2, 3, 4, 5}, 1.0f, false})
    ->ARGS_E2->Arg(1000);

BENCHMARK_TEMPLATE1_CAPTURE(BM_int_scan, x86_avx512_512_scan, E2_ColumnsCXL0_4Blade,
                            {{1}, {1}, {2, 3, 4, 5}, .0f, false})
    ->ARGS_E2->Arg(1);
BENCHMARK_TEMPLATE1_CAPTURE(BM_int_scan, x86_avx512_512_scan, E2_ColumnsCXL10_4Blade,
                            {{1}, {1}, {2, 3, 4, 5}, .1f, false})
    ->ARGS_E2->Arg(1);
BENCHMARK_TEMPLATE1_CAPTURE(BM_int_scan, x86_avx512_512_scan, E2_ColumnsCXL20_4Blade,
                            {{1}, {1}, {2, 3, 4, 5}, .2f, false})
    ->ARGS_E2->Arg(1);
BENCHMARK_TEMPLATE1_CAPTURE(BM_int_scan, x86_avx512_512_scan, E2_ColumnsCXL30_4Blade,
                            {{1}, {1}, {2, 3, 4, 5}, .3f, false})
    ->ARGS_E2->Arg(1);
BENCHMARK_TEMPLATE1_CAPTURE(BM_int_scan, x86_avx512_512_scan, E2_ColumnsCXL40_4Blade,
                            {{1}, {1}, {2, 3, 4, 5}, .4f, false})
    ->ARGS_E2->Arg(1);
BENCHMARK_TEMPLATE1_CAPTURE(BM_int_scan, x86_avx512_512_scan, E2_ColumnsCXL50_4Blade,
                            {{1}, {1}, {2, 3, 4, 5}, .5f, false})
    ->ARGS_E2->Arg(1);
BENCHMARK_TEMPLATE1_CAPTURE(BM_int_scan, x86_avx512_512_scan, E2_ColumnsCXL60_4Blade,
                            {{1}, {1}, {2, 3, 4, 5}, .6f, false})
    ->ARGS_E2->Arg(1);
BENCHMARK_TEMPLATE1_CAPTURE(BM_int_scan, x86_avx512_512_scan, E2_ColumnsCXL70_4Blade,
                            {{1}, {1}, {2, 3, 4, 5}, .7f, false})
    ->ARGS_E2->Arg(1);
BENCHMARK_TEMPLATE1_CAPTURE(BM_int_scan, x86_avx512_512_scan, E2_ColumnsCXL80_4Blade,
                            {{1}, {1}, {2, 3, 4, 5}, .8f, false})
    ->ARGS_E2->Arg(1);
BENCHMARK_TEMPLATE1_CAPTURE(BM_int_scan, x86_avx512_512_scan, E2_ColumnsCXL90_4Blade,
                            {{1}, {1}, {2, 3, 4, 5}, .9f, false})
    ->ARGS_E2->Arg(1);
BENCHMARK_TEMPLATE1_CAPTURE(BM_int_scan, x86_avx512_512_scan, E2_ColumnsCXL100_4Blade,
                            {{1}, {1}, {2, 3, 4, 5}, 1.0f, false})
    ->ARGS_E2->Arg(1);

// Experiment E3 - share of columns on CXL, TIDs on CXL
BENCHMARK_TEMPLATE1_CAPTURE(BM_int_scan, x86_avx512_512_scan, E3_ColumnsCXL0_1Blade, {{1}, {1}, {2}, .0f, true})
    ->ARGS_E2->Arg(1000);
BENCHMARK_TEMPLATE1_CAPTURE(BM_int_scan, x86_avx512_512_scan, E3_ColumnsCXL10_1Blade, {{1}, {1}, {2}, .1f, true})
    ->ARGS_E2->Arg(1000);
BENCHMARK_TEMPLATE1_CAPTURE(BM_int_scan, x86_avx512_512_scan, E3_ColumnsCXL20_1Blade, {{1}, {1}, {2}, .2f, true})
    ->ARGS_E2->Arg(1000);
BENCHMARK_TEMPLATE1_CAPTURE(BM_int_scan, x86_avx512_512_scan, E3_ColumnsCXL30_1Blade, {{1}, {1}, {2}, .3f, true})
    ->ARGS_E2->Arg(1000);
BENCHMARK_TEMPLATE1_CAPTURE(BM_int_scan, x86_avx512_512_scan, E3_ColumnsCXL40_1Blade, {{1}, {1}, {2}, .4f, true})
    ->ARGS_E2->Arg(1000);
BENCHMARK_TEMPLATE1_CAPTURE(BM_int_scan, x86_avx512_512_scan, E3_ColumnsCXL50_1Blade, {{1}, {1}, {2}, .5f, true})
    ->ARGS_E2->Arg(1000);
BENCHMARK_TEMPLATE1_CAPTURE(BM_int_scan, x86_avx512_512_scan, E3_ColumnsCXL60_1Blade, {{1}, {1}, {2}, .6f, true})
    ->ARGS_E2->Arg(1000);
BENCHMARK_TEMPLATE1_CAPTURE(BM_int_scan, x86_avx512_512_scan, E3_ColumnsCXL70_1Blade, {{1}, {1}, {2}, .7f, true})
    ->ARGS_E2->Arg(1000);
BENCHMARK_TEMPLATE1_CAPTURE(BM_int_scan, x86_avx512_512_scan, E3_ColumnsCXL80_1Blade, {{1}, {1}, {2}, .8f, true})
    ->ARGS_E2->Arg(1000);
BENCHMARK_TEMPLATE1_CAPTURE(BM_int_scan, x86_avx512_512_scan, E3_ColumnsCXL90_1Blade, {{1}, {1}, {2}, .9f, true})
    ->ARGS_E2->Arg(1000);
BENCHMARK_TEMPLATE1_CAPTURE(BM_int_scan, x86_avx512_512_scan, E3_ColumnsCXL100_1Blade, {{1}, {1}, {2}, 1.0f, true})
    ->ARGS_E2->Arg(1000);

BENCHMARK_TEMPLATE1_CAPTURE(BM_int_scan, x86_avx512_512_scan, E3_ColumnsCXL0_1Blade, {{1}, {1}, {2}, .0f, true})
    ->ARGS_E2->Arg(1);
BENCHMARK_TEMPLATE1_CAPTURE(BM_int_scan, x86_avx512_512_scan, E3_ColumnsCXL10_1Blade, {{1}, {1}, {2}, .1f, true})
    ->ARGS_E2->Arg(1);
BENCHMARK_TEMPLATE1_CAPTURE(BM_int_scan, x86_avx512_512_scan, E3_ColumnsCXL20_1Blade, {{1}, {1}, {2}, .2f, true})
    ->ARGS_E2->Arg(1);
BENCHMARK_TEMPLATE1_CAPTURE(BM_int_scan, x86_avx512_512_scan, E3_ColumnsCXL30_1Blade, {{1}, {1}, {2}, .3f, true})
    ->ARGS_E2->Arg(1);
BENCHMARK_TEMPLATE1_CAPTURE(BM_int_scan, x86_avx512_512_scan, E3_ColumnsCXL40_1Blade, {{1}, {1}, {2}, .4f, true})
    ->ARGS_E2->Arg(1);
BENCHMARK_TEMPLATE1_CAPTURE(BM_int_scan, x86_avx512_512_scan, E3_ColumnsCXL50_1Blade, {{1}, {1}, {2}, .5f, true})
    ->ARGS_E2->Arg(1);
BENCHMARK_TEMPLATE1_CAPTURE(BM_int_scan, x86_avx512_512_scan, E3_ColumnsCXL60_1Blade, {{1}, {1}, {2}, .6f, true})
    ->ARGS_E2->Arg(1);
BENCHMARK_TEMPLATE1_CAPTURE(BM_int_scan, x86_avx512_512_scan, E3_ColumnsCXL70_1Blade, {{1}, {1}, {2}, .7f, true})
    ->ARGS_E2->Arg(1);
BENCHMARK_TEMPLATE1_CAPTURE(BM_int_scan, x86_avx512_512_scan, E3_ColumnsCXL80_1Blade, {{1}, {1}, {2}, .8f, true})
    ->ARGS_E2->Arg(1);
BENCHMARK_TEMPLATE1_CAPTURE(BM_int_scan, x86_avx512_512_scan, E3_ColumnsCXL90_1Blade, {{1}, {1}, {2}, .9f, true})
    ->ARGS_E2->Arg(1);
BENCHMARK_TEMPLATE1_CAPTURE(BM_int_scan, x86_avx512_512_scan, E3_ColumnsCXL100_1Blade, {{1}, {1}, {2}, 1.0f, true})
    ->ARGS_E2->Arg(1);

BENCHMARK_TEMPLATE1_CAPTURE(BM_int_scan, x86_avx512_512_scan, E3_ColumnsCXL0_4Blade,
                            {{1}, {1}, {2, 3, 4, 5}, .0f, true})
    ->ARGS_E2->Arg(1000);
BENCHMARK_TEMPLATE1_CAPTURE(BM_int_scan, x86_avx512_512_scan, E3_ColumnsCXL10_4Blade,
                            {{1}, {1}, {2, 3, 4, 5}, .1f, true})
    ->ARGS_E2->Arg(1000);
BENCHMARK_TEMPLATE1_CAPTURE(BM_int_scan, x86_avx512_512_scan, E3_ColumnsCXL20_4Blade,
                            {{1}, {1}, {2, 3, 4, 5}, .2f, true})
    ->ARGS_E2->Arg(1000);
BENCHMARK_TEMPLATE1_CAPTURE(BM_int_scan, x86_avx512_512_scan, E3_ColumnsCXL30_4Blade,
                            {{1}, {1}, {2, 3, 4, 5}, .3f, true})
    ->ARGS_E2->Arg(1000);
BENCHMARK_TEMPLATE1_CAPTURE(BM_int_scan, x86_avx512_512_scan, E3_ColumnsCXL40_4Blade,
                            {{1}, {1}, {2, 3, 4, 5}, .4f, true})
    ->ARGS_E2->Arg(1000);
BENCHMARK_TEMPLATE1_CAPTURE(BM_int_scan, x86_avx512_512_scan, E3_ColumnsCXL50_4Blade,
                            {{1}, {1}, {2, 3, 4, 5}, .5f, true})
    ->ARGS_E2->Arg(1000);
BENCHMARK_TEMPLATE1_CAPTURE(BM_int_scan, x86_avx512_512_scan, E3_ColumnsCXL60_4Blade,
                            {{1}, {1}, {2, 3, 4, 5}, .6f, true})
    ->ARGS_E2->Arg(1000);
BENCHMARK_TEMPLATE1_CAPTURE(BM_int_scan, x86_avx512_512_scan, E3_ColumnsCXL70_4Blade,
                            {{1}, {1}, {2, 3, 4, 5}, .7f, true})
    ->ARGS_E2->Arg(1000);
BENCHMARK_TEMPLATE1_CAPTURE(BM_int_scan, x86_avx512_512_scan, E3_ColumnsCXL80_4Blade,
                            {{1}, {1}, {2, 3, 4, 5}, .8f, true})
    ->ARGS_E2->Arg(1000);
BENCHMARK_TEMPLATE1_CAPTURE(BM_int_scan, x86_avx512_512_scan, E3_ColumnsCXL90_4Blade,
                            {{1}, {1}, {2, 3, 4, 5}, .9f, true})
    ->ARGS_E2->Arg(1000);
BENCHMARK_TEMPLATE1_CAPTURE(BM_int_scan, x86_avx512_512_scan, E3_ColumnsCXL100_4Blade,
                            {{1}, {1}, {2, 3, 4, 5}, 1.0f, true})
    ->ARGS_E2->Arg(1000);

BENCHMARK_TEMPLATE1_CAPTURE(BM_int_scan, x86_avx512_512_scan, E3_ColumnsCXL0_4Blade,
                            {{1}, {1}, {2, 3, 4, 5}, .0f, true})
    ->ARGS_E2->Arg(1);
BENCHMARK_TEMPLATE1_CAPTURE(BM_int_scan, x86_avx512_512_scan, E3_ColumnsCXL10_4Blade,
                            {{1}, {1}, {2, 3, 4, 5}, .1f, true})
    ->ARGS_E2->Arg(1);
BENCHMARK_TEMPLATE1_CAPTURE(BM_int_scan, x86_avx512_512_scan, E3_ColumnsCXL20_4Blade,
                            {{1}, {1}, {2, 3, 4, 5}, .2f, true})
    ->ARGS_E2->Arg(1);
BENCHMARK_TEMPLATE1_CAPTURE(BM_int_scan, x86_avx512_512_scan, E3_ColumnsCXL30_4Blade,
                            {{1}, {1}, {2, 3, 4, 5}, .3f, true})
    ->ARGS_E2->Arg(1);
BENCHMARK_TEMPLATE1_CAPTURE(BM_int_scan, x86_avx512_512_scan, E3_ColumnsCXL40_4Blade,
                            {{1}, {1}, {2, 3, 4, 5}, .4f, true})
    ->ARGS_E2->Arg(1);
BENCHMARK_TEMPLATE1_CAPTURE(BM_int_scan, x86_avx512_512_scan, E3_ColumnsCXL50_4Blade,
                            {{1}, {1}, {2, 3, 4, 5}, .5f, true})
    ->ARGS_E2->Arg(1);
BENCHMARK_TEMPLATE1_CAPTURE(BM_int_scan, x86_avx512_512_scan, E3_ColumnsCXL60_4Blade,
                            {{1}, {1}, {2, 3, 4, 5}, .6f, true})
    ->ARGS_E2->Arg(1);
BENCHMARK_TEMPLATE1_CAPTURE(BM_int_scan, x86_avx512_512_scan, E3_ColumnsCXL70_4Blade,
                            {{1}, {1}, {2, 3, 4, 5}, .7f, true})
    ->ARGS_E2->Arg(1);
BENCHMARK_TEMPLATE1_CAPTURE(BM_int_scan, x86_avx512_512_scan, E3_ColumnsCXL80_4Blade,
                            {{1}, {1}, {2, 3, 4, 5}, .8f, true})
    ->ARGS_E2->Arg(1);
BENCHMARK_TEMPLATE1_CAPTURE(BM_int_scan, x86_avx512_512_scan, E3_ColumnsCXL90_4Blade,
                            {{1}, {1}, {2, 3, 4, 5}, .9f, true})
    ->ARGS_E2->Arg(1);
BENCHMARK_TEMPLATE1_CAPTURE(BM_int_scan, x86_avx512_512_scan, E3_ColumnsCXL100_4Blade,
                            {{1}, {1}, {2, 3, 4, 5}, 1.0f, true})
    ->ARGS_E2->Arg(1);

#define ARGS_E1A_100 Unit(benchmark::kMicrosecond)->Iterations(1)->Threads(24)->UseRealTime()->Arg(1000)

#define ARGS_E1A_1 Unit(benchmark::kMicrosecond)->Iterations(1)->Threads(24)->UseRealTime()->Arg(1)

// Experiment Analysis E1A
// E1A_100
BENCHMARK_TEMPLATE1_CAPTURE(BM_int_scan, x86_avx512_512_scan, E1A_100_AllLocal, {{1}, {1}, {INVALID_NODE}, 0.0f, false})
    ->ARGS_E1A_100;
BENCHMARK_TEMPLATE1_CAPTURE(BM_int_scan, x86_avx512_512_scan, E1A_100_ColumnsCXL1Blade, {{1}, {1}, {2}, 1.0f, false})
    ->ARGS_E1A_100;
BENCHMARK_TEMPLATE1_CAPTURE(BM_int_scan, x86_avx512_512_scan, E1A_100_ColumnsCXL4Blades,
                            {{1}, {1}, {2, 3, 4, 5}, 1.0f, false})
    ->ARGS_E1A_100;
BENCHMARK_TEMPLATE1_CAPTURE(BM_int_scan, x86_avx512_512_scan, E1A_100_AllCXL1Blade, {{1}, {1}, {2}, 1.0f, true})
    ->ARGS_E1A_100;
BENCHMARK_TEMPLATE1_CAPTURE(BM_int_scan, x86_avx512_512_scan, E1A_100_AllCXL4Blades,
                            {{1}, {1}, {2, 3, 4, 5}, 1.0f, true})
    ->ARGS_E1A_100;
// E1A_1
BENCHMARK_TEMPLATE1_CAPTURE(BM_int_scan, x86_avx512_512_scan, E1A_1_AllLocal, {{1}, {1}, {INVALID_NODE}, 0.0f, false})
    ->ARGS_E1A_1;
BENCHMARK_TEMPLATE1_CAPTURE(BM_int_scan, x86_avx512_512_scan, E1A_1_ColumnsCXL1Blade, {{1}, {1}, {2}, 1.0f, false})
    ->ARGS_E1A_1;
BENCHMARK_TEMPLATE1_CAPTURE(BM_int_scan, x86_avx512_512_scan, E1A_1_ColumnsCXL4Blades,
                            {{1}, {1}, {2, 3, 4, 5}, 1.0f, false})
    ->ARGS_E1A_1;
BENCHMARK_TEMPLATE1_CAPTURE(BM_int_scan, x86_avx512_512_scan, E1A_1_AllCXL1Blade, {{1}, {1}, {2}, 1.0f, true})
    ->ARGS_E1A_1;
BENCHMARK_TEMPLATE1_CAPTURE(BM_int_scan, x86_avx512_512_scan, E1A_1_AllCXL4Blades, {{1}, {1}, {2, 3, 4, 5}, 1.0f, true})
    ->ARGS_E1A_1;
#endif

BENCHMARK_MAIN();
