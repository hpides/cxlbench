#include "utils.hpp"

#include <fcntl.h>
#include <numa.h>
#include <sched.h>
#include <spdlog/spdlog.h>
#include <unistd.h>

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <random>
#include <thread>

#include "json.hpp"
#include "numa.hpp"
#include "read_write_ops.hpp"

namespace mema::utils {

char* map(const size_t expected_length, const bool use_huge_pages, const NumaNodeIDs& numa_memory_nodes) {
  // Do not mmap any data if length is 0
  if (expected_length == 0) {
    return nullptr;
  }

  void* addr = mmap(nullptr, expected_length, PROT_READ | PROT_WRITE, MAP_FLAGS, -1, 0);
  if (addr == MAP_FAILED || addr == nullptr) {
    spdlog::critical("Could not map anonymous memory region. Error: {}", std::strerror(errno));
    crash_exit();
  }

  // If no memory nodes are specified, the system will use the 'local allocation' policy, see
  // https://docs.kernel.org/admin-guide/mm/numa_memory_policy.html
  if (!numa_memory_nodes.empty()) {
    set_memory_on_numa_nodes(addr, expected_length, numa_memory_nodes);

    auto nodes_to_string = [](const auto nodes) {
      auto ss = std::stringstream{};
      if (!nodes.empty()) {
        std::copy(nodes.begin(), nodes.end() - 1, std::ostream_iterator<NumaNodeID>(ss, ", "));
        ss << nodes.back();
      }
      return ss.str();
    };
    spdlog::debug("Finished binding memory region of size {} to NUMA nodes {}.", expected_length,
                  nodes_to_string(numa_memory_nodes));
  } else {
    spdlog::debug("Did not bind memory to NUMA nodes: no NUMA nodes given.");
  }

  if (use_huge_pages) {
    if (madvise(addr, expected_length, MADV_HUGEPAGE) == -1) {
      spdlog::critical("madavise for huge pages failed. Error: {}", std::strerror(errno));
      crash_exit();
    } else {
      spdlog::debug("Enabled Transparent Huge Pages for the given memory region.");
    }
  } else {
    // Explicitly don't use huge pages.
    if (madvise(addr, expected_length, MADV_NOHUGEPAGE) == -1) {
      spdlog::critical("madavise for no huge pages failed. Error: {}", std::strerror(errno));
      crash_exit();
    } else {
      spdlog::debug("Prohibited Transparent Huge Pages for the given memory region.");
    }
  }

  return static_cast<char*>(addr);
}

// Returns node IDs on which the current thread is allowed to run on.
NumaNodeIDs get_numa_task_nodes() {
  auto max_node_id = numa_max_node();
  auto run_nodes = NumaNodeIDs{};

  // Check which NUMA node the calling thread is allowed to run on.

  const auto* const run_node_mask = numa_get_run_node_mask();

  for (auto node_id = NumaNodeID{0}; node_id <= max_node_id; ++node_id) {
    if (numa_bitmask_isbitset(run_node_mask, node_id)) {
      run_nodes.emplace_back(node_id);
    }
  }

  if (!run_nodes.empty()) {
    return run_nodes;
  }

  spdlog::critical("Could not determine NUMA task nodes of calling thread.");
  crash_exit();

  // This is only here to silence the "control reaches end of non-void function"
  // compiler warning. They were observed on GCC 12 and Clang 15.
  return {};
}

void generate_read_data(char* addr, const uint64_t memory_size) {
  if (memory_size == 0) {
    spdlog::debug("Did not generate data as the memory size was 0.");
    return;
  }

  spdlog::debug("Generating {} GB of random data to read.", memory_size / ONE_GB);
  std::vector<std::thread> thread_pool;
  thread_pool.reserve(DATA_GEN_THREAD_COUNT - 1);
  uint64_t thread_memory_size = memory_size / DATA_GEN_THREAD_COUNT;
  for (uint8_t thread_count = 0; thread_count < DATA_GEN_THREAD_COUNT - 1; thread_count++) {
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

void prefault_memory(char* addr, const uint64_t memory_size, const uint64_t page_size) {
  if (memory_size == 0) {
    spdlog::debug("Did not prefault the memory region as the memory size was 0.");
    return;
  }

  spdlog::debug("Pre-faulting data.");
  const size_t num_prefault_pages = memory_size / page_size;
  for (size_t prefault_offset = 0; prefault_offset < num_prefault_pages; ++prefault_offset) {
    addr[prefault_offset * page_size] = '\0';
  }
}

uint64_t duration_to_nanoseconds(const std::chrono::steady_clock::duration duration) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();
}

// FROM https://www.csee.usf.edu/~kchriste/tools/genzipf.c and
// https://stackoverflow.com/questions/9983239/how-to-generate-zipf-distributed-numbers-efficiently
//===========================================================================
//=  Function to generate Zipf (power law) distributed random variables     =
//=    - Input: alpha and N                                                 =
//=    - Output: Returns with Zipf distributed random variable              =
//===========================================================================
uint64_t zipf(const double alpha, const uint64_t n) {
  static thread_local bool first = true;  // Static first time flag
  static thread_local double c = 0;       // Normalization constant
  static thread_local double* sum_probs;  // Pre-calculated sum of probabilities
  double z;                               // Uniform random number (0 < z < 1)
  int zipf_value;                         // Computed exponential value to be returned

  // Compute normalization constant on first call only
  if (first) {
    for (uint64_t i = 1; i <= n; i++) {
      c = c + (1.0 / pow(static_cast<double>(i), alpha));
    }
    c = 1.0 / c;

    sum_probs = static_cast<double*>(malloc((n + 1) * sizeof(*sum_probs)));
    sum_probs[0] = 0;
    for (uint64_t i = 1; i <= n; i++) {
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
  const int64_t a = 16807;       // Multiplier
  const int64_t m = 2147483647;  // Modulus
  const int64_t q = 127773;      // m div a
  const int64_t r = 2836;        // m mod a
  static int64_t x = 1687248;    // Random int value
  int64_t x_div_q;               // x divided by q
  int64_t x_mod_q;               // x modulo q
  int64_t x_new;                 // New x value

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

std::string get_time_string() {
  auto now = std::chrono::system_clock::now();
  auto in_time_t = std::chrono::system_clock::to_time_t(now);
  std::stringstream ss;
  ss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d-%H-%M-%S");
  return ss.str();
}

std::filesystem::path create_result_file(const std::filesystem::path& result_dir,
                                         const std::filesystem::path& config_path) {
  std::error_code ec;
  const bool created = std::filesystem::create_directories(result_dir, ec);
  if (!created && ec) {
    spdlog::critical("Could not create result directory! Error: {}", ec.message());
    utils::crash_exit();
  }

  std::string file_name;
  const std::string result_suffix = "-results-" + get_time_string() + ".json";
  if (std::filesystem::is_regular_file(config_path)) {
    file_name = config_path.stem().concat(result_suffix);
  } else if (std::filesystem::is_directory(config_path)) {
    std::filesystem::path config_dir_name = *(--config_path.end());
    file_name = config_dir_name.concat(result_suffix);
  } else {
    spdlog::critical("Unexpected config file type for '{}'.", config_path.string());
    utils::crash_exit();
  }

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
