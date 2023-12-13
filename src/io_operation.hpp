#pragma once

#include <thread>
#include <vector>

#include "benchmark_config.hpp"
#include "fast_random.hpp"
#include "read_write_ops.hpp"
#include "spdlog/spdlog.h"
#include "utils.hpp"

namespace mema {

class IoOperation {
  friend class Benchmark;

 public:
  IoOperation(std::vector<char*>&& op_addresses, uint32_t access_size, Operation op_type,
              FlushInstruction flush_instruction)
      : op_addresses_{std::move(op_addresses)},
        access_size_{access_size},
        op_type_{op_type},
        flush_instruction_{flush_instruction} {}

  IoOperation() : IoOperation{{}, 0, Operation::Read, FlushInstruction::None} {};

  IoOperation(const IoOperation&) = delete;
  IoOperation& operator=(const IoOperation&) = delete;
  IoOperation(IoOperation&&) = default;
  IoOperation& operator=(IoOperation&&) = default;
  ~IoOperation() = default;

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
      case 64:
        return rw_ops::read_64(op_addresses_);
      case 128:
        return rw_ops::read_128(op_addresses_);
      case 256:
        return rw_ops::read_256(op_addresses_);
      case 512:
        return rw_ops::read_512(op_addresses_);
      case 1024:
        return rw_ops::read_1k(op_addresses_);
      case 2048:
        return rw_ops::read_2k(op_addresses_);
      case 4096:
        return rw_ops::read_4k(op_addresses_);
      case 8192:
        return rw_ops::read_8k(op_addresses_);
      case 16384:
        return rw_ops::read_16k(op_addresses_);
      case 32768:
        return rw_ops::read_32k(op_addresses_);
      case 65536:
        return rw_ops::read_64k(op_addresses_);
      default:
        return rw_ops::read(op_addresses_, access_size_);
    }
  }

  void run_write() {
    switch (flush_instruction_) {
      case FlushInstruction::Cache: {
#ifdef HAS_CLWB
        switch (access_size_) {
          case 64:
            return rw_ops::write_clwb_64(op_addresses_);
          case 128:
            return rw_ops::write_clwb_128(op_addresses_);
          case 256:
            return rw_ops::write_clwb_256(op_addresses_);
          case 512:
            return rw_ops::write_clwb_512(op_addresses_);
          case 1024:
            return rw_ops::write_clwb_1k(op_addresses_);
          case 2048:
            return rw_ops::write_clwb_2k(op_addresses_);
          case 4096:
            return rw_ops::write_clwb_4k(op_addresses_);
          case 8192:
            return rw_ops::write_clwb_8k(op_addresses_);
          case 16384:
            return rw_ops::write_clwb_16k(op_addresses_);
          case 32768:
            return rw_ops::write_clwb_32k(op_addresses_);
          case 65536:
            return rw_ops::write_clwb_64k(op_addresses_);
          default:
            return rw_ops::write_clwb(op_addresses_, access_size_);
        }
#else
        spdlog::critical("Compiled without CLWB support.");
        utils::crash_exit();
#endif  // HAS_CLWB
      }
      case FlushInstruction::NoCache: {
#if defined(USE_AVX_2) || defined(USE_AVX_512)
        switch (access_size_) {
          case 64:
            return rw_ops::simd_write_nt_64(op_addresses_);
          case 128:
            return rw_ops::simd_write_nt_128(op_addresses_);
          case 256:
            return rw_ops::simd_write_nt_256(op_addresses_);
          case 512:
            return rw_ops::simd_write_nt_512(op_addresses_);
          case 1024:
            return rw_ops::simd_write_nt_1k(op_addresses_);
          case 2048:
            return rw_ops::simd_write_nt_2k(op_addresses_);
          case 4096:
            return rw_ops::simd_write_nt_4k(op_addresses_);
          case 8192:
            return rw_ops::simd_write_nt_8k(op_addresses_);
          case 16384:
            return rw_ops::simd_write_nt_16k(op_addresses_);
          case 32768:
            return rw_ops::simd_write_nt_32k(op_addresses_);
          case 65536:
            return rw_ops::simd_write_nt_64k(op_addresses_);
          default:
            return rw_ops::simd_write_nt(op_addresses_, access_size_);
        }
#else
        spdlog::critical("Compiled without NT store support.");
        utils::crash_exit();
#endif  // NT stores
      }
      case FlushInstruction::None: {
        switch (access_size_) {
          case 64:
            return rw_ops::write_none_64(op_addresses_);
          case 128:
            return rw_ops::write_none_128(op_addresses_);
          case 256:
            return rw_ops::write_none_256(op_addresses_);
          case 512:
            return rw_ops::write_none_512(op_addresses_);
          case 1024:
            return rw_ops::write_none_1k(op_addresses_);
          case 2048:
            return rw_ops::write_none_2k(op_addresses_);
          case 4096:
            return rw_ops::write_none_4k(op_addresses_);
          case 8192:
            return rw_ops::write_none_8k(op_addresses_);
          case 16384:
            return rw_ops::write_none_16k(op_addresses_);
          case 32768:
            return rw_ops::write_none_32k(op_addresses_);
          case 65536:
            return rw_ops::write_none_64k(op_addresses_);
          default:
            return rw_ops::write_none(op_addresses_, access_size_);
        }
      }
    }
  }

  std::vector<char*> op_addresses_;
  uint32_t access_size_;
  Operation op_type_;
  FlushInstruction flush_instruction_;
};

class ChainedOperation {
 public:
  ChainedOperation(const CustomOp& op, char* range_start, const size_t range_size)
      : range_start_(range_start),
        access_size_(op.size),
        range_size_(range_size),
        align_(-access_size_),
        type_(op.type),
        flush_instruction_(op.flush),
        offset_(op.offset) {}

  inline void run(char* current_addr, char* dependent_addr) {
    if (type_ == Operation::Read) {
      current_addr = get_random_address(dependent_addr);
      dependent_addr = run_read(current_addr);
    } else {
      current_addr += offset_;
      run_write(current_addr);
    }

    if (next_) {
      return next_->run(current_addr, dependent_addr);
    }
  }

  inline char* get_random_address(char* addr) {
    const uint64_t base = (uint64_t)addr;
    const uint64_t random_offset = base + lehmer64();
    const uint64_t offset_in_range = random_offset % range_size_;
    const uint64_t aligned_offset = offset_in_range & align_;
    return range_start_ + aligned_offset;
  }

  void set_next(ChainedOperation* next) { next_ = next; }

 private:
  inline char* run_read(char* addr) {
    mema::rw_ops::CharVec read_value{};
    switch (access_size_) {
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
        read_value = rw_ops::read(addr, access_size_);
    }

    return reinterpret_cast<char*>(read_value[0]);
  }

  inline void run_write(char* addr) {
    switch (flush_instruction_) {
      case FlushInstruction::Cache: {
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
            return rw_ops::write_clwb(addr, access_size_);
        }
#else
        spdlog::critical("Compiled without CLWB support.");
        utils::crash_exit();
#endif  // HAS_CLWB
      }
      case FlushInstruction::NoCache: {
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
            return rw_ops::simd_write_nt(addr, access_size_);
        }
#else
        spdlog::critical("Compiled without NT store support.");
        utils::crash_exit();
#endif  // NT stores
      }
      case FlushInstruction::None: {
        switch (access_size_) {
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
            return rw_ops::write_none(addr, access_size_);
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
  const FlushInstruction flush_instruction_;
  const int64_t offset_;
};

}  // namespace mema
