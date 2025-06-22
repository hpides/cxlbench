#pragma once

#include <thread>
#include <vector>

#include "benchmark_config.hpp"
#include "fast_random.hpp"
#include "read_write_ops.hpp"
#include "spdlog/spdlog.h"
#include "utils.hpp"

namespace mema {

class AccessBatch {
  friend class Benchmark;

 public:
  AccessBatch(std::vector<char*>&& op_addresses, u32 access_size, Operation op_type, CacheInstruction cache_instruction)
      : addresses_{std::move(op_addresses)},
        access_size_{access_size},
        op_type_{op_type},
        cache_instruction_{cache_instruction} {}

  AccessBatch() : AccessBatch{{}, 0, Operation::Read, CacheInstruction::None} {};

  AccessBatch(const AccessBatch&) = delete;
  AccessBatch& operator=(const AccessBatch&) = delete;
  AccessBatch(AccessBatch&&) = default;
  AccessBatch& operator=(AccessBatch&&) = default;
  ~AccessBatch() = default;

  inline void run() {
    switch (op_type_) {
      case Operation::Read: {
        return run_read();
      }
      case Operation::Write: {
        return run_write();
      }
      default: {
        spdlog::critical("Invalid operation: {}", op_type_);
        utils::crash_exit();
      }
    }
  }

  inline bool is_read() const { return op_type_ == Operation::Read; }
  inline bool is_write() const { return op_type_ == Operation::Write; }

 private:
  void run_read() {
    switch (access_size_) {
      case 4:
        return rw_ops::read_4(addresses_);
      case 8:
        return rw_ops::read_8(addresses_);
      case 16:
        return rw_ops::read_16(addresses_);
      case 32:
        return rw_ops::read_32(addresses_);
      case 64:
        return rw_ops::read_64(addresses_);
      case 128:
        return rw_ops::read_128(addresses_);
      case 256:
        return rw_ops::read_256(addresses_);
      case 512:
        return rw_ops::read_512(addresses_);
      case 1024:
        return rw_ops::read_1k(addresses_);
      case 2048:
        return rw_ops::read_2k(addresses_);
      case 4096:
        return rw_ops::read_4k(addresses_);
      case 8192:
        return rw_ops::read_8k(addresses_);
      case 16384:
        return rw_ops::read_16k(addresses_);
      case 32768:
        return rw_ops::read_32k(addresses_);
      case 65536:
        return rw_ops::read_64k(addresses_);
      default:
        spdlog::critical("Access size not supported.");
        utils::crash_exit();
    }
  }

  void run_write() {
    switch (cache_instruction_) {
      case CacheInstruction::Cache: {
#ifdef HAS_CLWB
        switch (access_size_) {
          case 64:
            return rw_ops::write_clwb_64(addresses_);
          case 128:
            return rw_ops::write_clwb_128(addresses_);
          case 256:
            return rw_ops::write_clwb_256(addresses_);
          case 512:
            return rw_ops::write_clwb_512(addresses_);
          case 1024:
            return rw_ops::write_clwb_1k(addresses_);
          case 2048:
            return rw_ops::write_clwb_2k(addresses_);
          case 4096:
            return rw_ops::write_clwb_4k(addresses_);
          case 8192:
            return rw_ops::write_clwb_8k(addresses_);
          case 16384:
            return rw_ops::write_clwb_16k(addresses_);
          case 32768:
            return rw_ops::write_clwb_32k(addresses_);
          case 65536:
            return rw_ops::write_clwb_64k(addresses_);
          default:
            spdlog::critical("Access size not supported.");
            utils::crash_exit();
        }
#else
        spdlog::critical("Compiled without CLWB support.");
        utils::crash_exit();
#endif  // HAS_CLWB
      }
      case CacheInstruction::NoCache: {
#if defined(USE_AVX_2) || defined(USE_AVX_512)
        switch (access_size_) {
          case 64:
            return rw_ops::simd_write_nt_64(addresses_);
          case 128:
            return rw_ops::simd_write_nt_128(addresses_);
          case 256:
            return rw_ops::simd_write_nt_256(addresses_);
          case 512:
            return rw_ops::simd_write_nt_512(addresses_);
          case 1024:
            return rw_ops::simd_write_nt_1k(addresses_);
          case 2048:
            return rw_ops::simd_write_nt_2k(addresses_);
          case 4096:
            return rw_ops::simd_write_nt_4k(addresses_);
          case 8192:
            return rw_ops::simd_write_nt_8k(addresses_);
          case 16384:
            return rw_ops::simd_write_nt_16k(addresses_);
          case 32768:
            return rw_ops::simd_write_nt_32k(addresses_);
          case 65536:
            return rw_ops::simd_write_nt_64k(addresses_);
          default:
            spdlog::critical("Access size not supported.");
            utils::crash_exit();
        }
#else
        spdlog::critical("Compiled without NT store support.");
        utils::crash_exit();
#endif  // NT stores
      }
      case CacheInstruction::None: {
        switch (access_size_) {
          case 4:
            return rw_ops::write_none_4(addresses_);
          case 8:
            return rw_ops::write_none_8(addresses_);
          case 16:
            return rw_ops::write_none_16(addresses_);
          case 32:
            return rw_ops::write_none_32(addresses_);
          case 64:
            return rw_ops::write_none_64(addresses_);
          case 128:
            return rw_ops::write_none_128(addresses_);
          case 256:
            return rw_ops::write_none_256(addresses_);
          case 512:
            return rw_ops::write_none_512(addresses_);
          case 1024:
            return rw_ops::write_none_1k(addresses_);
          case 2048:
            return rw_ops::write_none_2k(addresses_);
          case 4096:
            return rw_ops::write_none_4k(addresses_);
          case 8192:
            return rw_ops::write_none_8k(addresses_);
          case 16384:
            return rw_ops::write_none_16k(addresses_);
          case 32768:
            return rw_ops::write_none_32k(addresses_);
          case 65536:
            return rw_ops::write_none_64k(addresses_);
          default:
            spdlog::critical("Access size not supported.");
            utils::crash_exit();
        }
      }
    }
  }

  std::vector<char*> addresses_;
  u32 access_size_;
  Operation op_type_;
  CacheInstruction cache_instruction_;
};

class ChainedOperation {
 public:
  ChainedOperation(const CustomOp& op, char* range_start, const size_t range_size)
      : range_start_(range_start),
        access_size_(op.size),
        range_size_(range_size),
        align_(-access_size_),
        type_(op.type),
        cache_instruction_(op.cache_fn),
        offset_(op.offset) {}

  inline void run(char* current_addr, char dependent_value) {
    if (type_ == Operation::Read) {
      current_addr = get_random_address(dependent_value);
      dependent_value = run_read(current_addr);
    } else {
      current_addr += offset_;
      run_write(current_addr);
    }

    if (next_) {
      return next_->run(current_addr, dependent_value);
    }
  }

  inline char* get_random_address(char dependent_value) {
    const u64 value = (u64)dependent_value;
    const u64 random_offset = value + lehmer64();
    const u64 offset_in_range = random_offset % range_size_;
    const u64 aligned_offset = offset_in_range & align_;
    return range_start_ + aligned_offset;
  }

  void set_next(ChainedOperation* next) { next_ = next; }

 private:
  inline char run_read(char* addr) {
    char read_value{};
    switch (access_size_) {
      case 4:
        read_value = rw_ops::read_4(addr);
        break;
      case 8:
        read_value = rw_ops::read_8(addr);
        break;
      case 16:
        read_value = rw_ops::read_16(addr);
        break;
      case 32:
        read_value = rw_ops::read_32(addr);
        break;
      case 64:
        read_value = rw_ops::read_64(addr);
        break;
      case 128:
        read_value = rw_ops::read_128(addr);
        break;
      case 256:
        read_value = rw_ops::read_256(addr);
        break;
      case 512:
        read_value = rw_ops::read_512(addr);
        break;
      case 1024:
        read_value = rw_ops::read_1k(addr);
        break;
      case 2048:
        read_value = rw_ops::read_2k(addr);
        break;
      case 4096:
        read_value = rw_ops::read_4k(addr);
        break;
      case 8192:
        read_value = rw_ops::read_8k(addr);
        break;
      case 16384:
        read_value = rw_ops::read_16k(addr);
        break;
      case 32768:
        read_value = rw_ops::read_32k(addr);
        break;
      case 65536:
        read_value = rw_ops::read_64k(addr);
        break;
      default:
        spdlog::critical("Access size not supported.");
        utils::crash_exit();
    }

    return read_value;
  }

  inline void run_write(char* addr) {
    switch (cache_instruction_) {
      case CacheInstruction::Cache: {
#ifdef HAS_CLWB
        switch (access_size_) {
          case 64:
            return rw_ops::write_clwb_64(addr);
          case 128:
            return rw_ops::write_clwb_128(addr);
          case 256:
            return rw_ops::write_clwb_256(addr);
          case 512:
            return rw_ops::write_clwb_512(addr);
          case 1024:
            return rw_ops::write_clwb_1k(addr);
          case 2048:
            return rw_ops::write_clwb_2k(addr);
          case 4096:
            return rw_ops::write_clwb_4k(addr);
          case 8192:
            return rw_ops::write_clwb_8k(addr);
          case 16384:
            return rw_ops::write_clwb_16k(addr);
          case 32768:
            return rw_ops::write_clwb_32k(addr);
          case 65536:
            return rw_ops::write_clwb_64k(addr);
          default:
            spdlog::critical("Access size not supported.");
            utils::crash_exit();
        }
#else
        spdlog::critical("Compiled without CLWB support.");
        utils::crash_exit();
#endif  // HAS_CLWB
      }
      case CacheInstruction::NoCache: {
#if defined(USE_AVX_2) || defined(USE_AVX_512)
        switch (access_size_) {
          case 64:
            return rw_ops::simd_write_nt_64(addr);
          case 128:
            return rw_ops::simd_write_nt_128(addr);
          case 256:
            return rw_ops::simd_write_nt_256(addr);
          case 512:
            return rw_ops::simd_write_nt_512(addr);
          case 1024:
            return rw_ops::simd_write_nt_1k(addr);
          case 2048:
            return rw_ops::simd_write_nt_2k(addr);
          case 4096:
            return rw_ops::simd_write_nt_4k(addr);
          case 8192:
            return rw_ops::simd_write_nt_8k(addr);
          case 16384:
            return rw_ops::simd_write_nt_16k(addr);
          case 32768:
            return rw_ops::simd_write_nt_32k(addr);
          case 65536:
            return rw_ops::simd_write_nt_64k(addr);
          default:
            spdlog::critical("Access size not supported.");
            utils::crash_exit();
        }
#else
        spdlog::critical("Compiled without NT store support.");
        utils::crash_exit();
#endif  // NT stores
      }
      case CacheInstruction::None: {
        switch (access_size_) {
          case 4:
            return rw_ops::write_none_4(addr);
          case 8:
            return rw_ops::write_none_8(addr);
          case 16:
            return rw_ops::write_none_16(addr);
          case 32:
            return rw_ops::write_none_32(addr);
          case 64:
            return rw_ops::write_none_64(addr);
          case 128:
            return rw_ops::write_none_128(addr);
          case 256:
            return rw_ops::write_none_256(addr);
          case 512:
            return rw_ops::write_none_512(addr);
          case 1024:
            return rw_ops::write_none_1k(addr);
          case 2048:
            return rw_ops::write_none_2k(addr);
          case 4096:
            return rw_ops::write_none_4k(addr);
          case 8192:
            return rw_ops::write_none_8k(addr);
          case 16384:
            return rw_ops::write_none_16k(addr);
          case 32768:
            return rw_ops::write_none_32k(addr);
          case 65536:
            return rw_ops::write_none_64k(addr);
          default:
            spdlog::critical("Access size not supported.");
            utils::crash_exit();
        }
      }
    }
  }

 private:
  char* const range_start_;
  const size_t access_size_;
  const size_t range_size_;
  const size_t align_;
  ChainedOperation* next_ = nullptr;
  const Operation type_;
  const CacheInstruction cache_instruction_;
  const i64 offset_;
};

}  // namespace mema
