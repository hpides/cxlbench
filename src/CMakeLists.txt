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
        access_batch.hpp
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
        threads.cpp
        threads.hpp
        types.hpp
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

# Microbenchmark cacheline
add_executable(bench-cacheline microexperiments/bench_cacheline.cpp)
target_link_libraries(bench-cacheline benchmark::benchmark ${NUMA_LIBRARIES})
set_target_properties(bench-cacheline PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

# Microbenchmark largher than node allocations
add_executable(larger-than-node-allocations microexperiments/larger_than_node_allocation.cpp)
target_link_libraries(larger-than-node-allocations benchmark::benchmark ${NUMA_LIBRARIES})
set_target_properties(larger-than-node-allocations PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

# Tool to check move_pages
add_executable(check_move_pages microexperiments/check_move_pages.cpp)
target_link_libraries(check_move_pages ${NUMA_LIBRARIES})
set_target_properties(check_move_pages PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

# Microbenchmark false sharing (Ulrich Drepper)
add_executable(false-sharing-drepper microexperiments/falsesharing_drepper/main.cpp)
target_link_libraries(false-sharing-drepper)
set_target_properties(false-sharing-drepper PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

# Microbenchmark false sharing (based on Drepper with modern C++)
add_executable(false-sharing microexperiments/falsesharing/main.cpp)
target_link_libraries(false-sharing)
set_target_properties(false-sharing PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

# Playground
add_executable(playground playground/main.cpp)
target_link_libraries(playground mema)
set_target_properties(playground PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

# Clang tidy
if(MEMATIDY)
  set(CMAKE_CXX_CLANG_TIDY "clang-tidy")
  set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
  message(STATUS "Enabled clang-tidy")
  set_target_properties(false-sharing PROPERTIES CXX_CLANG_TIDY "clang-tidy")
endif()

include(CMakePrintHelpers)
cmake_print_variables(CMAKE_CXX_FLAGS)
