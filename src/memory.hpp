#pragma once

#include <fcntl.h>
#include <sys/mman.h>

#include <cstddef>
#include <cstring>

#include "types.hpp"

namespace cxlbench {

int open_device(const std::string& device_path) {
  const auto file_descriptor = open(device_path.c_str(), O_RDWR | O_SYNC);
  BenchAssert(file_descriptor != -1, "Failed open device");
  return file_descriptor;
}

inline std::byte* map_private_populate(size_t size, int file_descriptor, size_t offset) {
  auto* buffer = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_POPULATE, file_descriptor, offset);
  BenchAssert(buffer != MAP_FAILED, std::string{"mmap failed: "} + std::strerror(errno));
  return static_cast<std::byte*>(buffer);
}

inline std::byte* map_shared_populate(size_t size, int file_descriptor, size_t offset) {
  auto* buffer = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, file_descriptor, offset);
  BenchAssert(buffer != MAP_FAILED, std::string{"mmap failed: "} + std::strerror(errno));
  return static_cast<std::byte*>(buffer);
}

inline std::byte* map_private_anonymous_populate(size_t size, int file_descriptor, size_t offset) {
  auto* buffer =
      mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE, file_descriptor, offset);
  BenchAssert(buffer != MAP_FAILED, std::string{"mmap failed: "} + std::strerror(errno));
  return static_cast<std::byte*>(buffer);
}

inline std::byte* map_shared_anonymous_populate(size_t size, int file_descriptor, size_t offset) {
  auto* buffer =
      mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS | MAP_POPULATE, file_descriptor, offset);
  BenchAssert(buffer != MAP_FAILED, std::string{"mmap failed: "} + std::strerror(errno));
  return static_cast<std::byte*>(buffer);
}

inline void unmap(std::byte* buffer, size_t size) { munmap(static_cast<void*>(buffer), size); }

}  // namespace cxlbench
