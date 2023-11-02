set(
        SOURCES

        benchmark.cpp
        benchmark.hpp
        benchmark_config.cpp
        benchmark_config.hpp
        benchmark_factory.cpp
        benchmark_factory.hpp
        benchmark_suite.cpp
        benchmark_suite.hpp
        fast_random.cpp
        fast_random.hpp
        io_operation.hpp
        numa.cpp
        numa.hpp
        single_benchmark.cpp
        single_benchmark.hpp
        parallel_benchmark.cpp
        parallel_benchmark.hpp
        read_write_ops_avx2.hpp
        read_write_ops_avx512.hpp
        read_write_ops_types.hpp
        read_write_ops.hpp
        utils.cpp
        utils.hpp
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${COMPILE_FLAGS}")

add_library(mema STATIC ${SOURCES})
target_link_libraries(mema benchmark::benchmark hdr_histogram_static spdlog stdc++fs yaml-cpp)

target_link_libraries(mema ${NUMA_LIBRARIES})

add_executable(mema-bench mema.cpp)
target_link_libraries(mema-bench mema)
set_target_properties(mema-bench PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

# Some third party libraries' include directories lead to compiler warnings. We add these directories add system
# directories to supporess the warnings.
# TODO(MW) suppress shadow warnings.
get_target_property(yaml-cpp_include_dirs yaml-cpp INTERFACE_INCLUDE_DIRECTORIES)
target_include_directories(yaml-cpp SYSTEM PUBLIC ${yaml-cpp_include_dirs})

# Microbenchmark targets
add_executable(bench-cacheline microexperiments/bench_cacheline.cpp)
target_link_libraries(bench-cacheline benchmark::benchmark ${NUMA_LIBRARIES})
set_target_properties(bench-cacheline PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

add_executable(larger-than-node-allocations microexperiments/larger_than_node_allocation.cpp)
target_link_libraries(larger-than-node-allocations benchmark::benchmark ${NUMA_LIBRARIES})
set_target_properties(larger-than-node-allocations PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
