#include <fcntl.h>      // open
#include <immintrin.h>  // clwb
#include <string.h>
#include <sys/mman.h>  // mmap
#include <unistd.h>    // getpagesize

#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <thread>
#include <unordered_map>

// NOLINTBEGIN
#define BenchAssert(expr, msg)       \
  if (!static_cast<bool>(expr)) {    \
    std::cerr << (msg) << std::endl; \
  }                                  \
  static_assert(true, "End call of macro with a semicolon")
// NOLINTEND

// GNR: dax0.0, 00-04-cf-fc-03-ff-fe-09 <--+          3000000000-12fffffffff
// EMR: dax0.0, 00-04-cf-fc-03-ff-fe-0e    | shared    8080000000-1807fffffff
// EMR: dax1.0, 00-04-cf-fc-03-ff-fe-10    |          18080000000-2807fffffff
// EMR: dax2.0, 00-04-cf-fc-03-ff-fe-08 <--+          28080000000-3807fffffff
// EMR: dax3.0, 00-04-cf-fc-03-ff-fe-1a               38080000000-4807fffffff

// Minimum example, taken from https://github.com/cxl-reskit/cxl-reskit#using-cxl-memory-as-a-dax-device
int minimum_example() {
  /* DAX mapping requires a 2MiB alignment */
  uint64_t page_size = 2 * 1024 * 1024;

  int fd = open("/dev/dax0.0", O_RDWR);
  if (fd == -1) {
    perror("open() failed");
    return 1;
  }

  void* dax_addr = mmap(nullptr, page_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (dax_addr == MAP_FAILED) {
    perror("mmap() failed");
    close(fd);
    return 1;
  }

  /* Write something to the memory */
  strcpy((char*)dax_addr, "hello world");

  munmap(dax_addr, page_size);
  close(fd);
  return 0;
}

inline void print(char* buffer, const uint32_t size) {
  for (int i = 0; i < size; ++i) {
    std::cout << buffer[i];
  }
  std::cout << " (addr: " << static_cast<const void*>(buffer) << ")" << std::endl;
}

inline void copy_from_cin(char* buffer) {
  std::cout << "copy" << std::endl;
  for (auto i = uint32_t{0}; i < 64; ++i) {
    auto read_char = char{};
    if (i == 64 || !(std::cin.get(read_char)) || read_char == '\n') {
      return;
    }
    std::cout << "write at " << i << std::endl;
    buffer[i] = read_char;
  }
}

char* memory_region(const std::string& system) {
  /* DAX mapping requires a 2MiB alignment */
  uint64_t page_size = 2 * 1024 * 1024;
  static auto cxl_device = std::unordered_map<std::string, const char*>{};
  cxl_device["GNR"] = "/dev/dax0.0";
  cxl_device["EMR"] = "/dev/dax2.0";

  const auto& device_name = cxl_device[system];
  auto memfd = open(device_name, O_RDWR | O_SYNC);
  if (memfd == -1) {
    fprintf(stderr, "failed to open %s for physical memory: %s\n", device_name, strerror(errno));
    exit(1);
  }
  printf("open retuned: %X\n", memfd);
  const auto size = 1024u;
  const auto aligned_size = (size + page_size - 1) & ~(page_size - 1);

  const auto offset = 0u;
  std::cout << device_name << ", offset: " << offset << std::endl;
  auto buffer = mmap(0, aligned_size, PROT_READ | PROT_WRITE, MAP_SHARED, memfd, offset);
  if (buffer == MAP_FAILED) {
    perror("mmap");
    close(memfd);
    exit(1);
  }
  return reinterpret_cast<char*>(buffer);
}

void write(char* buffer) {
  auto input = std::string{};
  while (true) {
    std::getline(std::cin, input);
    std::memcpy(buffer, input.c_str(), input.size());
    _mm_clwb(buffer);
    _mm_sfence();
    print(buffer, 64);
  }
}

void read(char* buffer) {
  while (true) {
    _mm_clflush(buffer);
    print(buffer, 64);
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
  }
}

int main(int argc, char* argv[]) {
  //  BenchAssert(argc >= 3, "usage: <binary> (-r|-w) <dax idx>");
  BenchAssert(argc >= 3, "usage: <binary> (-r|-w) (EMR|GNR)");
  auto is_read = [&]() {
    auto option = std::string{argv[1]};
    if (option == "-r") {
      return true;
    }
    if (option == "-w") {
      return false;
    }
    throw std::logic_error{"wrong argument, needs to be -r or -w"};
  }();

  auto system = std::string{argv[2]};
  BenchAssert(system == "EMR" || system == "GNR", "System needs to be 'EMR' or 'GNR'.");

  std::cout << "is read: " << is_read << ", system: " << system << std::endl;
  auto* buffer = memory_region(system);
  std::cout << "memory region done" << std::endl;
  if (is_read) {
    read(buffer);
  }
  write(buffer);

  return 0;
}
