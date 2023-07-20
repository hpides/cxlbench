![cmake + test](https://github.com/mweisgut/mema-bench/actions/workflows/cmake.yml/badge.svg) ![clang format](https://github.com/mweisgut/mema-bench/actions/workflows/clang-format.yml/badge.svg)

# MemA-Bench

A benchmarking suite and toolset to evaluate the performance of persistent memory access.

## Quick Start
MemA-Bench has a predefined set of workloads to benchmark and requires very little configuration to run directly on
your system.
If you have the development version of [libnuma](https://github.com/numactl/numactl) (MemA-Bench requires the headers)
installed in the default locations, e.g., via `apt install libnuma-dev`, you can simply run the commands below.
Otherwise, you should briefly check out our [Build Options](#build-options) beforehand.

```shell script
$ git clone git@github.com:mweisgut/mema-bench.git
$ cd mema-bench
$ mkdir build && cd build
$ cmake .. -DBUILD_TEST=ON -DCMAKE_C_COMPILER=gcc-12 -DCMAKE_CXX_COMPILER=g++-12 -DCMAKE_BUILD_TYPE=Release
$ make -j
$ ./mema-bench
```

This will create a `results` directory containing a JSON file with all benchmark results in it.

### Build Options
In the following, we describe which build options you can provide for MemA-Bench and how to configure them.

#### Using libnuma
In order to get NUMA-awareness in the benchmarks, you should have `libnuma` installed in the default location, e.g.,
via `apt install libnuma-dev` or `yum install numactl-devel`.
If you have `libnuma` installed at a different location, you can specify `-DNUMA_INCLUDE_PATH` and `-DNUMA_LIBRARY_PATH`
to point to the respective headers and library in the `cmake` command.

#### Building tests
By default, MemA-Bench will not build the tests.
If you want to run the tests to make sure that everything was built correctly, you can specify `-DBUILD_TEST=ON` to
build the tests.
This is mainly relevant for development though.

## Configuring Benchmarks
You can specify custom benchmarks via YAML files.
For examples, please check out the files in the [workloads](workloads/) directory.
Each benchmark consists of a `matrix` and an `args` part.
In the `matrix`, you specify arguments that you want to iterate through.
For example, if you specify `number_threads: [1, 4, 8]` and `access_size: [64, 128]`, you will get 3x2 = 6 benchmarks, each a variant as specified by the cross product of both lists.
In the `args` part, you specify the arguments that are the same across all matrix runs.

An example:
```yaml
random_reads:
  # This will generate six benchmarks, with the thread/access combinations:
  # (1, 64), (1, 128), (4, 64), (4, 128), (8, 64), (8, 128)
  matrix:
    number_threads: [ 1, 4, 8 ]
    access_size: [ 64, 128 ]

  # Each of the six runs will perform 200 million random reads on a 10 GiB memory range.
  args:
    operation: read
    memory_range: 10G
    number_operations: 200000000
    exec_mode: random
```

We currently support the following options (with default values).
This code is taken from [src/benchmark_config.hpp](src/benchmark_config.hpp).
```cpp
/** Represents the size of an individual memory access in Byte. Must be a power of two. */
uint32_t access_size = 256;

/** Represents the total PMem memory range to use for the benchmark. Must be a multiple of `access_size`.  */
uint64_t memory_range = 10 * BYTES_IN_GIGABYTE;  // 10 GiB

/** Represents the total DRAM memory range to use for the benchmark. Must be a multiple of `access_size`.  */
uint64_t dram_memory_range = 0;

/** Represents the ratio of DRAM IOOperations to PMem IOOperations. Must only contain one digit after decimal point,
 * i.e., 0.1 or 0.2. */
double dram_operation_ratio = 0.0;

/** Represents the number of random access / custom operations to perform. Can *not* be set for sequential access. */
uint64_t number_operations = 100'000'000;

/** Number of threads to run the benchmark with. Must be a power of two. */
uint16_t number_threads = 1;

/** Alternative measure to end a benchmark by letting is run for `run_time` seconds. */
uint64_t run_time = 0;

/** Type of memory access operation to perform, i.e., read or write. */
Operation operation = Operation::Read;

/** Mode of execution, i.e., sequential, random, or custom. See `Mode` for all options. */
Mode exec_mode = Mode::Sequential;

/** Persist instruction to use after write operations. Only works with `Operation::Write`. See
 * `PersistInstruction` for more details on available options. */
PersistInstruction persist_instruction = PersistInstruction::NoCache;

/** Number of disjoint memory regions to partition the `memory_range` into. Must be 0 or a divisor of
 * `number_threads` i.e., one or more threads map to one partition. When set to 0, it is equal to the number of
 * threads, i.e., each thread has its own partition. Default is set to 1.  */
uint16_t number_partitions = 1;

/** Specifies the set of memory NUMA nodes on which benchmark data is to be allocated. */
NumaNodeIDs numa_memory_nodes;

/** Distribution to use for `Mode::Random`, i.e., uniform of zipfian. */
RandomDistribution random_distribution = RandomDistribution::Uniform;

/** Zipf skew factor for `Mode::Random` and `RandomDistribution::Zipf`. */
double zipf_alpha = 0.9;

/** List of custom operations to use in `Mode::Custom`. See `CustomOp` for more details on string representation.  */
std::vector<CustomOp> custom_operations;

/** Frequency in which to sample latency of custom operations. Only works in combination with `Mode::Custom`. */
uint64_t latency_sample_frequency = 0;

/** Whether or not to prefault the memory region before writing to it. If set to false, the benchmark will include the
 * time caused by page faults on first access to the allocated memory region. */
bool prefault_file = true;

/** Whether or not to use transparent huge pages in DRAM, i.e., 2 MiB instead of regular 4 KiB pages. */
bool dram_huge_pages = true;

/** Represents the minimum size of an atomic work package. A chunk contains chunk_size / access_size number of
 * operations. Assuming the lowest bandwidth of 1 GiB/s operations per thread, 64 MiB is a ~60 ms execution unit. */
uint64_t min_io_chunk_size = 64 * BYTES_IN_MEGABYTE;
```

## Running Custom Memory Access Patterns
Beside the standard workloads (sequential/random read/write), you can also specify more complex access pattern.
These can be used to represent, e.g., data structure access.
You can check out the examples in [workloads/operations/](workloads/operations/).
To use these custom workloads, you need to specify them as `custom_operations` in the YAML and choose `exec_mode: custom`.

The string representation of a custom operation is:
For reads: `r(<location>)_<size>`
with:
 'r' for read,
 (optional) `<location>` is 'd' or 'p' for DRAM/PMem (with p as default is nothing is specified),
 `<size>` is the size of the access (must be power of 2).

For writes: `w(<location>)_<size>_<persist_instruction>(_<offset>)`
with:
 'w' for write,
 (optional) `<location>` is 'd' or 'p' for DRAM/PMem (with p as default is nothing is specified),
 `<size>` is the size of the access (must be power of 2),
 `<persist_instruction>` is the instruction to use after the write (none, cache, cacheinv, noache),
 (optional) `<offset>` is the offset to the previously accessed address (can be negative, default is 0)

See the following example for more details.

```yaml
# Represent updates to a (very simplified) hybrid tree-index structure.
hybrid_tree_index_update:
  matrix:
    # ...
  args:
    # This string represents a random 1024 Byte read to DRAM (rd_1024), followed by a dependent (pointer-chasing) 1024 Byte read to PMem (rp_1024). It the performs a 64 Byte write 512 Bytes into the previously read 1024 Byte (wp_64_cache_512), followed by the same write at the beginning pf the initial 1024 Byte (specified via _-512).
    custom_operations: "rd_1024,rp_1024,wp_64_cache_512,wp_64_cache_-512"

    # As we use both DRAM and PMem, we must also specify a DRAM range.
    dram_memory_range: 10G
    memory_range: 20G

    exec_mode: custom
    # ...
```
## Used AVX-512 Intrinsics

Read: `_mm512_load_si512`
```
extern __m512i __cdecl _mm512_load_si512(void const* mem_addr);
```

>Load 512-bits of integer data from memory into destination.
>
>mem_addr must be aligned on a 64-byte boundary or a general-protection exception will be generated.

Write Non-Temporal: `_mm512_stream_si512`
```
extern void __cdecl _mm512_stream_si512(void* mem_addr, __m512i a);
```
>Store 512-bits of integer data from a into memory using a non-temporal memory hint.

Write: `_mm512_store_si512`
```
extern void __cdecl _mm512_store_si512(void* mem_addr, __m512i a);
```
>Store 512-bits of integer data from a into memory.
>
>mem_addr must be aligned on a 64-byte boundary or a general-protection exception will be generated.

Based on Intel's documentation [Intrinsics for Integer Load and Store Operations](https://www.intel.com/content/www/us/en/docs/cpp-compiler/developer-guide-reference/2021-8/intrinsics-for-integer-load-and-store-operations.html)
