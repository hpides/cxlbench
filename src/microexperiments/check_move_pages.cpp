#include <numa.h>
#include <numaif.h>
#include <stdint.h>
#include <sys/mman.h>

#include <array>
#include <cstring>
#include <iostream>
#include <limits>
#include <vector>

namespace {

#define Assert(expr, msg)         \
  if (!static_cast<bool>(expr)) { \
    throw std::logic_error{msg};  \
  }                               \
  static_assert(true, "End call of macro with a semicolon")

constexpr auto KiB = uint64_t{1024};
constexpr auto MiB = KiB * 1024;
constexpr auto PAGE_SIZE = 4 * KiB;
constexpr auto NODES_SIZE = 2;

}  // namespace

int main(int argc, char** argv) {
  // check args
  if (argc != 4) {
    throw std::invalid_argument{"Need to specify <first node> <second node> <memory size in MiB>"};
  }

  const auto first_node = std::stoi(argv[1]);
  const auto second_node = std::stoi(argv[2]);
  const auto memory_size_in_mib = std::stoi(argv[3]);

  std::cout << "Nodes {" << first_node << ", " << second_node << "}, Size in MiB: " << memory_size_in_mib << std::endl;

  if (memory_size_in_mib == 0) {
    throw std::invalid_argument{"Memory size in MiB needs to be > 0."};
  }

  const auto memory_size = memory_size_in_mib * MiB;

  // create node array
  auto nodes = std::array<int32_t, NODES_SIZE>{first_node, second_node};

  // allocate
  auto addr = mmap(nullptr, memory_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

  if (addr == MAP_FAILED || addr == nullptr) {
    std::cout << "Could not map anonymous memory region. Error: " << std::strerror(errno) << "." << std::endl;
    return 1;
  }

  // mbind
  const auto* const allowed_memory_nodes_mask = numa_get_mems_allowed();
  auto* const nodemask = numa_allocate_nodemask();
  for (const auto node_id : nodes) {
    if (!numa_bitmask_isbitset(allowed_memory_nodes_mask, node_id)) {
      std::cout << "Memory allocation on numa node id not allowed (given: " << node_id << ")." << std::endl;
      return 1;
    }
    numa_bitmask_setbit(nodemask, node_id);
  }

  // Note that "[i]f the MPOL_INTERLEAVE policy was specified, pages already residing on the specified nodes will not be
  // moved such that they are interleaved", see mbind(2) manual. Therefore, we move pages manually via move_pages.
  if (nodemask == nullptr) {
    std::cout << "When setting the memory nodes, node mask cannot be nullptr." << std::endl;
  }
  const auto mbind_succeeded = mbind(addr, memory_size, MPOL_INTERLEAVE, nodemask->maskp, nodemask->size + 1,
                                     MPOL_MF_STRICT | MPOL_MF_MOVE) == 0;
  const auto mbind_errno = errno;
  numa_bitmask_free(nodemask);
  if (!mbind_succeeded) {
    std::cout << "mbind failed: " << strerror(mbind_errno) << "." << std::endl;
    return 1;
  }

  // populate
  auto* data = reinterpret_cast<char*>(addr);

  const auto page_count = memory_size / PAGE_SIZE;
  for (auto page_id = uint64_t{0}; page_id < page_count; ++page_id) {
    data[page_id * PAGE_SIZE] = '\0';
  }

  // // move_pages
  // auto page_target_nodes = std::vector<int>{};
  // {
  //   page_target_nodes.resize(page_count);
  //   auto pages = std::vector<void*>{};
  //   pages.resize(page_count);

  //   for (auto page_idx = uint64_t{0}; page_idx < page_count; ++page_idx) {
  //     page_target_nodes[page_idx] = nodes[page_idx % NODES_SIZE];
  //     pages[page_idx] = reinterpret_cast<void*>(data + page_idx * PAGE_SIZE);
  //   }

  //   auto page_status = std::vector<int>(page_count, std::numeric_limits<int>::max());
  //   const auto ret =
  //       move_pages(0, page_count, pages.data(), page_target_nodes.data(), page_status.data(), MPOL_MF_MOVE_ALL);
  //   const auto move_pages_errno = errno;

  //   if (ret != 0) {
  //     std::cout << "Placement: move_pages failed: " << strerror(move_pages_errno) << std::endl;
  //     return 1;
  //   }
  // }

  // verify
  {
    auto pages = std::vector<void*>{};
    pages.resize(page_count);

    for (auto page_idx = uint64_t{0}; page_idx < page_count; ++page_idx) {
      pages[page_idx] = reinterpret_cast<void*>(data + page_idx * PAGE_SIZE);
    }

    auto page_status = std::vector<int>(page_count, std::numeric_limits<int>::max());
    const auto ret = move_pages(0, page_count, pages.data(), NULL, page_status.data(), MPOL_MF_MOVE);
    const auto move_pages_errno = errno;

    if (ret != 0) {
      std::cout << "Verify: move_pages failed: " << strerror(move_pages_errno) << std::endl;
      return 1;
    }

    auto counter_first = 0u;
    auto counter_second = 0u;
    auto false_pages = 0u;
    for (auto page_idx = uint64_t{0}; page_idx < page_count; ++page_idx) {
      auto& status = page_status[page_idx];

      if (status == first_node) {
        ++counter_first;
        continue;
      }

      if (status == second_node) {
        ++counter_second;
        continue;
      }

      false_pages++;
      // auto& expected_status = page_target_nodes[page_idx];
      // if (status != expected_status) {
      //   false_pages++;
      //   if (page_idx % 100 == 0) {
      //     std::cout << "Page " << page_idx << " with status " << status << " but expected: " << expected_status
      //               << std::endl;
      //   }
      // }
    }
    std::cout << page_count << " pages, N" << first_node << ": " << counter_first << ", N" << second_node << ": "
              << counter_second << ", other status: " << false_pages << std::endl;
  }
}
