#! /usr/bin/env bash

# Exit the script immediately if an error occurs.
set -e

SCRIPTNAME="$(basename "$0")"
if [ -z "$1" ] || [ -z "$2" ] ; then
  echo "Usage: "$SCRIPTNAME" <build dir tag> <workload tag>"
  exit 1
fi

TAG="$1"
WORKLOAD="$2"

# Get the absolute path here.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$SCRIPT_DIR"/../ && pwd
BUILD_DIR_NAME=exp-"$TAG"-rel-gcc-12
BUILD_DIR="$ROOT_DIR"/"$BUILD_DIR_NAME"
mkdir -p "$BUILD_DIR"
echo "Build directory: \""$BUILD_DIR"\""
# Navigating to the build dir since `reset_workload.sh` needs to be executed from a build directory.
cd "$BUILD_DIR"

echo "Copying workload "$WORKLOAD" to the build dir's workload directory."
../scripts/reset_workload.sh "$WORKLOAD"

echo "CMake setup..."
CMAKE_CMD="cmake -S "$ROOT_DIR" -B "$BUILD_DIR" -DBUILD_TEST=ON -DCMAKE_C_COMPILER=gcc-12 -DCMAKE_CXX_COMPILER=g++-12 -DCMAKE_BUILD_TYPE=Release"
if echo "ninja version:" && ninja --version; then
  CMAKE_CMD="${CMAKE_CMD} -GNinja"
  BUILD_CMD="ninja"
  echo "Using ninja"
else
  BUILD_CMD="make -j"
  echo "Using make instead"
fi
eval "$CMAKE_CMD"
eval "$BUILD_CMD"

if [ "$3" == "-s" ]; then
  echo "Setting up the system."
  ../scripts/setup_system.sh
else
  echo "Not execution the system setup. If you want to run the system, add -s as the second option."
fi

START_TIME=$(eval date "+%FT%H-%M-%S-%N")
RESULT_DIR="$ROOT_DIR"/results/"$WORKLOAD"/"$TAG"/"$START_TIME"
mkdir -p "$RESULT_DIR"

# Run benchmarks
./mema-bench -r "$RESULT_DIR"
"$SCRIPT_DIR"/plot.sh "$RESULT_DIR"

exit 0
