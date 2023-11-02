![cmake + test](https://github.com/mweisgut/mema-bench/actions/workflows/cmake.yml/badge.svg) ![clang format](https://github.com/mweisgut/mema-bench/actions/workflows/clang-format.yml/badge.svg)

# MemA-Bench

A benchmarking suite and toolset to evaluate the performance of memory access.

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
    memory_region_size: 10G
    number_operations: 200000000
    exec_mode: random
```

Please check `src/benchmark_config.hpp` to see the supported benchmark configuration options.

## Running Custom Memory Access Patterns
Beside the standard workloads (sequential/random read/write), you can also specify more complex access pattern.
These can be used to represent, e.g., data structure access.
You can check out the examples in [workloads/operations/](workloads/operations/).
To use these custom workloads, you need to specify them as `custom_operations` in the YAML and choose `exec_mode: custom`.

The string representation of a custom operation is:
For reads: `r_<size>`
with:
 'r' for read,
 `<size>` is the size of the access (must be power of 2).

For writes: `w_<size>_<flush_instruction>(_<offset>)`
with:
 'w' for write,
 `<size>` is the size of the access (must be power of 2),
 `<flush_instruction>` is the instruction to use after the write (none, cache, noache),
 (optional) `<offset>` is the offset to the previously accessed address (can be negative, default is 0)

## System Configuration

For configuring your system for benchmark runs, you might want to use the `./scripts/system_setup.sh` script.
