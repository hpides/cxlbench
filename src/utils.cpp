#include "utils.hpp"

#include <fcntl.h>
#include <sched.h>
#include <spdlog/spdlog.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <random>
#include <thread>

#include "json.hpp"
#include "numa.hpp"
#include "read_write_ops.hpp"

namespace mema::utils {

char* map(const u64 expected_length, const bool use_transparent_huge_pages, const u64 explicit_hugepages_size) {
  spdlog::debug("Mapping memory region of size {}, explicit huge page size {}, use transparent huge pages {}.",
                expected_length, explicit_hugepages_size, use_transparent_huge_pages);
  // Do not mmap any data if length is 0
  if (expected_length == 0) {
    return nullptr;
  }

  void* addr = nullptr;
  if (explicit_hugepages_size > 0) {
    const auto page_size_mask = mmap_page_size_mask(explicit_hugepages_size);
    addr = mmap(nullptr, expected_length, PROT_READ | PROT_WRITE, MAP_FLAGS_HUGETLB | page_size_mask, -1, 0);
    spdlog::debug("Mapped memory region with explicit page size of {}.", explicit_hugepages_size);
  } else {
    addr = mmap(nullptr, expected_length, PROT_READ | PROT_WRITE, MAP_FLAGS, -1, 0);
    spdlog::debug("Mapped memory region without explicit page size.");
  }

  if (addr == MAP_FAILED || addr == nullptr) {
    spdlog::critical(
        "Could not map anonymous memory region. Error: {}. (1) If using explicit hugepages, ensure that you have "
        "enough "
        "pages allocated. (2) If using transparent huge pages, you might need to increase /proc/sys/vm/nr_hugepages. "
        "(3) Check if the allocation works with smaller memory regions.",
        std::strerror(errno));
    crash_exit();
  }

  if (!use_transparent_huge_pages) {
    // Explicitly don't use transparent huge pages.
    if (madvise(addr, expected_length, MADV_NOHUGEPAGE) == -1) {
      spdlog::critical("madavise for no huge pages failed. Error: {}", std::strerror(errno));
      crash_exit();
    } else {
      spdlog::debug("Prohibited Transparent Huge Pages for the given memory region.");
    }
  } else {
    if (madvise(addr, expected_length, MADV_HUGEPAGE) == -1) {
      spdlog::critical("madavise for huge pages failed. Error: {}", std::strerror(errno));
      crash_exit();
    } else {
      spdlog::debug("Enabled Transparent Huge Pages for the given memory region.");
    }
  }
  return static_cast<char*>(addr);
}

void populate_memory(char* addr, const u64 memory_size) {
  MemaAssert(memory_size % utils::PAGE_SIZE == 0, "Memory region needs to be a multiple of the page size.");
  if (memory_size == 0) {
    spdlog::warn("Did not populate/pre-fault the memory region as the memory size was 0.");
    return;
  }

  spdlog::debug("Populating/pre-faulting data.");
  const auto page_count = memory_size / utils::PAGE_SIZE;
  for (auto page_id = u64{0}; page_id < page_count; ++page_id) {
    addr[page_id * utils::PAGE_SIZE] = '\0';
  }
}

int mmap_page_size_mask(const u32 page_size) {
  if (((page_size) & (page_size - 1)) != 0) {
    spdlog::critical("Given page size {} is not a power of 2.", page_size);
    utils::crash_exit();
  }
  const auto required_bits = std::bit_width(page_size) - 1;
  spdlog::debug("Calculating mmap page size mask: {} bits required for {}", required_bits, page_size);
  return required_bits << MAP_HUGE_SHIFT;
}

void generate_shuffled_access_positions(char* addr, const MemoryRegionDefinition& region,
                                        const BenchmarkConfig& config) {
  auto buffer = reinterpret_cast<std::byte*>(addr);

  const auto batch_size = config.min_io_batch_size;
  const auto access_count_per_batch = batch_size / config.access_size;
  const auto batch_count = region.size / batch_size;

  std::random_device rd;
  std::mt19937 gen(rd());

  for (auto batch_id = u64{0}; batch_id < batch_count; ++batch_id) {
    auto indices = std::vector<u64>(access_count_per_batch);
    for (auto i = u64{0}; i < access_count_per_batch; ++i) {
      indices[i] = batch_id * access_count_per_batch + i;
    }
    std::shuffle(indices.begin(), indices.end(), gen);

    for (auto i = u64{0}; i < access_count_per_batch; ++i) {
      auto* position_addr = reinterpret_cast<u64*>(buffer + (indices[i] * config.access_size));
      *position_addr = indices[(i + 1) % access_count_per_batch];
    }
  }
}

bool verify_shuffled_access_positions(char* addr, const MemoryRegionDefinition& region, const BenchmarkConfig& config) {
  spdlog::info("Verify shuffled access positions.");
  auto buffer = reinterpret_cast<std::byte*>(addr);
  const auto total_entry_count = region.size / config.access_size;
  auto index_historgram = std::vector<u64>(total_entry_count, false);
  // Create histogram
  for (auto entry_idx = u64{0}; entry_idx < total_entry_count; ++entry_idx) {
    auto* index = reinterpret_cast<u64*>(&buffer[entry_idx * config.access_size]);
    const auto is_in_range = *index >= 0 && *index < total_entry_count;
    if (!is_in_range) {
      spdlog::critical("Access index at position {} with value {} is out of range [0, {})", entry_idx, *index,
                       total_entry_count);
      utils::crash_exit();
    }
    index_historgram[*index]++;
  }

  // Check the histogram values for missing values and duplicates
  auto verified = true;
  for (auto idx = u64{0}; idx < total_entry_count; ++idx) {
    // Missing index
    if (!index_historgram[idx]) {
      spdlog::warn("Access index {} is not present in shuffled access positions.", idx);
      verified = false;
      continue;
    }
    // Index duplicates
    if (index_historgram[idx] > 1) {
      spdlog::warn("Duplicates: access index {} is present {} times in shuffled access positions.", idx,
                   index_historgram[idx]);
      verified = false;
    }
  }
  return verified;
}

void generate_read_data(char* addr, const u64 memory_size) {
  if (memory_size == 0) {
    spdlog::debug("Did not generate data as the memory size was 0.");
    return;
  }

  spdlog::debug("Generating {} GB of random data to read.", memory_size / ONE_GIB);
  auto thread_pool = std::vector<std::thread>{};
  thread_pool.reserve(DATA_GEN_THREAD_COUNT - 1);
  auto thread_memory_size = memory_size / DATA_GEN_THREAD_COUNT;

  for (u8 thread_count = 0; thread_count < DATA_GEN_THREAD_COUNT - 1; thread_count++) {
    char* from = addr + thread_count * thread_memory_size;
    const char* to = addr + (thread_count + 1) * thread_memory_size;
    thread_pool.emplace_back(rw_ops::write_data, from, to);
  }

  // Since DATA_GEN_THREAD_COUNT - 1 already started writing data, we use the time to write the last partition of the
  // memory region.
  rw_ops::write_data(addr + (DATA_GEN_THREAD_COUNT - 1) * thread_memory_size, addr + memory_size);

  // wait for all threads
  for (std::thread& thread : thread_pool) {
    thread.join();
  }
  spdlog::debug("Finished generating data.");
}

void clear_caches() {
  // ~200 MB
  static constexpr auto value_count = 27000000;
  auto values = std::make_unique<std::array<u64, value_count>>();
  for (auto counter = u32{0}; auto& value : *values) {
    value = counter;
    ++counter;
  }
  benchmark::DoNotOptimize(values);
}

u64 duration_to_nanoseconds(const std::chrono::steady_clock::duration duration) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();
}

// FROM https://www.csee.usf.edu/~kchriste/tools/genzipf.c and
// https://stackoverflow.com/questions/9983239/how-to-generate-zipf-distributed-numbers-efficiently
//===========================================================================
//=  Function to generate Zipf (power law) distributed random variables     =
//=    - Input: alpha and N                                                 =
//=    - Output: Returns with Zipf distributed random variable              =
//===========================================================================
u64 zipf(const double alpha, const u64 n) {
  static thread_local bool first = true;  // Static first time flag
  static thread_local double c = 0;       // Normalization constant
  static thread_local double* sum_probs;  // Pre-calculated sum of probabilities
  double z;                               // Uniform random number (0 < z < 1)
  int zipf_value;                         // Computed exponential value to be returned

  // Compute normalization constant on first call only
  if (first) {
    for (u64 i = 1; i <= n; i++) {
      c = c + (1.0 / pow(static_cast<double>(i), alpha));
    }
    c = 1.0 / c;

    sum_probs = static_cast<double*>(malloc((n + 1) * sizeof(*sum_probs)));
    sum_probs[0] = 0;
    for (u64 i = 1; i <= n; i++) {
      sum_probs[i] = sum_probs[i - 1] + c / pow(static_cast<double>(i), alpha);
    }
    first = false;
  }

  // Pull a uniform random number (0 < z < 1)
  do {
    z = rand_val();
  } while ((z == 0) || (z == 1));

  // Map z to the value
  int low = 1, high = n, mid;
  do {
    mid = floor((low + high) / 2);
    if (sum_probs[mid] >= z && sum_probs[mid - 1] < z) {
      zipf_value = mid;
      break;
    } else if (sum_probs[mid] >= z) {
      high = mid - 1;
    } else {
      low = mid + 1;
    }
  } while (low <= high);

  return zipf_value - 1;  // Subtract one to map to a range from 0 - (n - 1)
}

//=========================================================================
//= Multiplicative LCG for generating uniform(0.0, 1.0) random numbers    =
//=   - x_n = 7^5*x_(n-1)mod(2^31 - 1)                                    =
//=   - With x seeded to 1 the 10000th x value should be 1043618065       =
//=   - From R. Jain, "The Art of Computer Systems Performance Analysis," =
//=     John Wiley & Sons, 1991. (Page 443, Figure 26.2)                  =
//=========================================================================
double rand_val() {
  const i64 a = 16807;       // Multiplier
  const i64 m = 2147483647;  // Modulus
  const i64 q = 127773;      // m div a
  const i64 r = 2836;        // m mod a
  static i64 x = 1687248;    // Random int value
  i64 x_div_q;               // x divided by q
  i64 x_mod_q;               // x modulo q
  i64 x_new;                 // New x value

  // RNG using integer arithmetic
  x_div_q = x / q;
  x_mod_q = x % q;
  x_new = (a * x_mod_q) - (r * x_div_q);
  if (x_new > 0) {
    x = x_new;
  } else {
    x = x_new + m;
  }

  // Return a random value between 0.0 and 1.0
  return (static_cast<double>(x) / m);
}

void crash_exit() { throw MemaException{}; }

void crash_exit(const std::string msg) { throw MemaException{msg}; }

std::string get_time_string() {
  auto now = std::chrono::system_clock::now();
  auto in_time_t = std::chrono::system_clock::to_time_t(now);
  std::stringstream ss;
  ss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d-%H-%M-%S");
  return ss.str();
}

std::string get_file_name_from_path(const std::filesystem::path& config_path, const std::string& file_extension) {
  auto stream = std::stringstream{};
  stream << "-" << get_time_string() << "." << file_extension;
  if (std::filesystem::is_regular_file(config_path)) {
    return config_path.stem().concat(stream.str());
  }

  if (!std::filesystem::is_directory(config_path)) {
    spdlog::critical("Unexpected config file type for '{}'.", config_path.string());
    utils::crash_exit();
  }

  std::filesystem::path config_dir_name = *(--config_path.end());
  if (config_dir_name.string().empty()) {
    return std::string{"result"} + stream.str();
  }
  return config_dir_name.concat(stream.str());
}

std::filesystem::path create_result_file(const std::filesystem::path& result_dir,
                                         const std::filesystem::path& config_path) {
  std::error_code ec;
  const bool created = std::filesystem::create_directories(result_dir, ec);
  if (!created && ec) {
    spdlog::critical("Could not create result directory! Error: {}", ec.message());
    utils::crash_exit();
  }

  auto file_name = get_file_name_from_path(config_path, "json");
  std::filesystem::path result_path = result_dir / file_name;
  std::ofstream result_file(result_path);
  result_file << nlohmann::json::array() << std::endl;

  return result_path;
}

void write_benchmark_results(const std::filesystem::path& result_path, const nlohmann::json& results) {
  nlohmann::json all_results;
  std::ifstream previous_result_file(result_path);
  previous_result_file >> all_results;

  if (!all_results.is_array()) {
    previous_result_file.close();
    spdlog::critical("Result file '{}' is corrupted! Content must be a valid JSON array.", result_path.string());
    utils::crash_exit();
  }

  all_results.push_back(results);
  // Clear all existing data.
  std::ofstream new_result_file(result_path, std::ofstream::trunc);
  new_result_file << std::setw(2) << all_results << std::endl;
}

void print_segfault_error() {
  spdlog::critical("A thread encountered an unexpected SIGSEGV!");
  spdlog::critical(
      "Please create an issue on GitHub (https://github.com/mweisgut/mema-bench/issues/new) "
      "with your configuration and system information so that we can try to fix this.");
}

}  // namespace mema::utils
