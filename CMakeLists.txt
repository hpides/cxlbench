cmake_minimum_required(VERSION 3.16)
project(mema-bench)

include(ExternalProject)
include(FetchContent)
include(CheckCXXSourceCompiles)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -pthread")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g3 -gdwarf-4 -fno-omit-frame-pointer -march=native -DNDEBUG" CACHE STRING "" FORCE)

# Required for ARM64
set(CMAKE_THREAD_LIBS_INIT "-pthread")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

if (${CMAKE_BUILD_TYPE} STREQUAL "Release")
    message("-- Release mode, all optimizations enabled")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -DNDEBUG")
endif ()

set(INCLUDE_HINTS ${CMAKE_INCLUDE_PATH} "/usr/include" "/usr/local/include")
set(LIB_HINTS "/usr/local/lib64" "/usr/local/lib" "/usr/lib" "/usr/lib64")

# CI
option(IS_CI_BUILD "Set true if this is a CI build.")

##################### AVX ####################
# Use this short program to check if AVX-512 is supported. For now, we only use AVX-512 or not.
# This might need to be adapted in the future to support different SSE/AVX instructions.
include(CheckCSourceRuns)
set(avx512_prog "int main() { asm volatile(\"vmovdqu64 %zmm0, %zmm1\"); return 0; }")
set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} -mavx512f")
check_c_source_runs("${avx512_prog}" HAS_AVX_512)

set(avx_prog "int main() { asm volatile(\"vpsllvq %ymm0, %ymm1, %ymm2\"); return 0; }")
set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} -mavx2")
check_c_source_runs("${avx_prog}" HAS_AVX_2)

function(fail_specify_avx)
  message(FATAL_ERROR "Please specify the instruction set to be used (only one):
  -DUSE_AVX_2=ON
  -DUSE_AVX_512=ON")
endfunction()

# In case of a CI build, we do not use AVX.
if (${IS_CI_BUILD})
    message(STATUS "CI build specified. Not using AVX.")
# Set HAS_AVX_512 or HAS_AVX_2 so we can use it to enable explicit AVX instructions.
elseif ((${HAS_AVX_512}) AND (${HAS_AVX_2}))
   if ((${USE_AVX_512}) AND (${USE_AVX_2}))
        fail_specify_avx()
   elseif (${USE_AVX_512})
        add_definitions("-DHAS_AVX_512" "-DHAS_ANY_AVX")
        message(STATUS "System supports AVX-2 and AVX-512, AVX-512 is used.")
   elseif (${USE_AVX_2})
        add_definitions("-DHAS_AVX_2" "-DHAS_ANY_AVX")
        message(STATUS "System supports AVX-2 and AVX-512, AVX-2 is used.")
   else()
        fail_specify_avx()
   endif()
elseif (${HAS_AVX_512})
    add_definitions("-DHAS_AVX_512" "-DHAS_ANY_AVX")
    message(STATUS "System supports AVX-512.")
elseif (${HAS_AVX_2})
    add_definitions("-DHAS_AVX_2" "-DHAS_ANY_AVX")
    message(STATUS "System supports AVX-2.")
else() 
    # No AVX instruction set supported.
    message(FATAL_ERROR "System does not support AVX-2 or AVX-512.")
endif()

# Backup of actual required flags
set(BASE_CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS}")

##################### CLWB ####################
set(clwb_prog "#include <immintrin.h>\n int main() { _mm_clwb((void*)0); return 0; }")
set(CMAKE_REQUIRED_FLAGS "${BASE_CMAKE_REQUIRED_FLAGS} -mclwb")
check_cxx_source_compiles("${clwb_prog}" HAS_CLWB)
if (${HAS_CLWB})
    # Set HAS_CLWB so we can use it to enable all explicit CLWB instructions.
    add_definitions("-DHAS_CLWB")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mclwb")
    message(STATUS "System supports CLWB.")
else ()
    message(STATUS "System does not support CLWB.")
endif ()

################### NUMA ####################
set(NUMA_INCLUDE_PATH "" CACHE STRING "Path to custom NUMA include files")
set(NUMA_LIBRARY_PATH "" CACHE STRING "Path to custom NUMA library")

set(NUMA_INCLUDE_HINTS ${NUMA_INCLUDE_PATH} ${INCLUDE_HINTS})
set(NUMA_LIB_HINTS ${NUMA_LIBRARY_PATH})

find_path(NUMA_INCLUDE_DIRS numa.h HINTS ${NUMA_INCLUDE_HINTS})
find_library(NUMA_LIBRARIES NAMES numa libnuma HINTS ${NUMA_LIB_HINTS})

set(NUMA_FOUND true)
if (NOT NUMA_INCLUDE_DIRS OR "${NUMA_INCLUDE_DIRS}" STREQUAL "")
    message(FATAL_ERROR "NUMA include directory not found.")
endif ()
if (NOT NUMA_LIBRARIES OR "${NUMA_LIBRARIES}" STREQUAL "")
    message(FATAL_ERROR "libnuma not found.")
endif ()

include_directories(${NUMA_INCLUDE_DIRS})
message(STATUS "NUMA provided. Including ${NUMA_INCLUDE_DIRS} and linking ${NUMA_LIBRARIES}.")

##################### YAML ####################
set(YAML_CPP_BUILD_TESTS OFF CACHE INTERNAL "")
set(YAML_CPP_INSTALL OFF CACHE INTERNAL "")
set(YAML_CPP_BUILD_TOOLS OFF CACHE INTERNAL "")

FetchContent_Declare(
        yaml_cpp
        GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
        GIT_TAG yaml-cpp-0.6.3
)
FetchContent_MakeAvailable(yaml_cpp)

##################### JSON ####################
set(JSON_VERSION v3.10.5)
set(JSON_DOWNLOAD_PATH https://github.com/nlohmann/json/releases/download/${JSON_VERSION}/json.hpp)
set(JSON_DIR ${CMAKE_CURRENT_BINARY_DIR}/json)
set(JSON_INCLUDE_PATH ${JSON_DIR}/json.hpp)
file(DOWNLOAD ${JSON_DOWNLOAD_PATH} ${JSON_INCLUDE_PATH})
include_directories(${JSON_DIR})

################### Hdr Histogram ####################
set(HDR_HISTOGRAM_BUILD_PROGRAMS OFF CACHE INTERNAL "")
FetchContent_Declare(
        hdr_histogram
        GIT_REPOSITORY https://github.com/HdrHistogram/HdrHistogram_c.git
        GIT_TAG 0.11.2
)
FetchContent_MakeAvailable(hdr_histogram)

################### spdlog ####################
FetchContent_Declare(
        spdlog
        GIT_REPOSITORY https://github.com/gabime/spdlog.git
        GIT_TAG v1.9.2
)
FetchContent_MakeAvailable(spdlog)

################### CLI11 ####################
set(CLI11_VERSION v2.1.2)
set(CLI11_DOWNLOAD_PATH https://github.com/CLIUtils/CLI11/releases/download/${CLI11_VERSION}/CLI11.hpp)
set(CLI11_DIR ${CMAKE_CURRENT_BINARY_DIR}/cli11)
set(CLI11_INCLUDE_PATH ${CLI11_DIR}/CLI11.hpp)
file(DOWNLOAD ${CLI11_DOWNLOAD_PATH} ${CLI11_INCLUDE_PATH})
include_directories(${CLI11_DIR})

##################### GoogleBenchmark ####################
set(BENCHMARK_ENABLE_TESTING OFF)
FetchContent_Declare(
        googlebenchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG        v1.7.1
)
FetchContent_MakeAvailable(googlebenchmark)

##################### MemA ####################
add_custom_target(copy_directory
        COMMAND ${CMAKE_COMMAND}
        -Dsource_dir=${CMAKE_SOURCE_DIR}/workloads
        -Ddestination_dir=${CMAKE_CURRENT_BINARY_DIR}/workloads
        -P ${CMAKE_SOURCE_DIR}/copy_directory.cmake
)
include_directories(src)
add_subdirectory(src)

##################### Test ####################
option(BUILD_TEST "Set true if tests should be built and run.")

if (${BUILD_TEST})
    message(STATUS "Tests are included in this build.")
    enable_testing()
    add_subdirectory(test)
endif ()
