#! /usr/bin/env bash

# Exit the script immediately if an error occurs.
set -e
BUILD_TYPE=Release

SCRIPTNAME="$(basename "$0")"
if [ -z "$1" ] || [ -z "$2" ] ; then
  echo "Usage: "$SCRIPTNAME" <build dir tag> <workload tag> [start_time]"
  exit 1
fi

TAG="$1"
WORKLOAD="$2"

if [ -z "$3" ]; then
  START_TIME=$(eval date "+%FT%H-%M-%S-%N")
  echo "Current time: $START_TIME"
else
  START_TIME="$3"
  echo "Given start time: $START_TIME"
fi

# Get the absolute path here.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$SCRIPT_DIR"/.. && pwd
BUILD_DIR_NAME=exp-"$TAG"-rel-gcc-12
BUILD_DIR="$ROOT_DIR"/"$BUILD_DIR_NAME"

echo "Workload: $2"
# Workload-specific flags
# if [[ "$2" =~ ^bw_expansion_.* ]]; then
#   WORKLOAD_FLAGS=-e
#   echo "Workload-specific flags: $WORKLOAD_FLAGS"
# fi

mkdir -p "$BUILD_DIR"
echo "Build directory: \""$BUILD_DIR"\""
# Navigating to the build dir since `reset_workload.sh` needs to be executed from a build directory.
cd "$BUILD_DIR"

echo "Copying workload "$WORKLOAD" to the build dir's workload directory."
../scripts/reset_workload.sh "$WORKLOAD"

echo "CMake setup..."
CMAKE_CMD="cmake -S "$ROOT_DIR" -B "$BUILD_DIR" -DBUILD_TEST=ON -DCMAKE_C_COMPILER=gcc-12 -DCMAKE_CXX_COMPILER=g++-12 -DCMAKE_BUILD_TYPE="$BUILD_TYPE
BUILD_CMD="make -j mema-bench"
echo "Using make instead"
eval "$CMAKE_CMD"
eval "$BUILD_CMD"

if [ "$3" == "-s" ]; then
  echo "Setting up the system."
  ../scripts/setup_system.sh
else
  echo "Not executing the system setup. If you want to run the system, add -s as the second option."
fi

RESULT_DIR="$ROOT_DIR"/results/"$WORKLOAD"/"$TAG"/"$START_TIME"
mkdir -p "$RESULT_DIR"

# Run benchmarks
./mema-bench -r $RESULT_DIR $WORKLOAD_FLAGS
echo "All results are written to $RESULT_DIR."
echo "If you want to copy the data to your dev machine, the following command might help:"
echo "./scripts/scp.sh $(hostname) $RESULT_DIR ./results/$WORKLOAD/$TAG"

exit 0
