#! /usr/bin/env bash

SCRIPTNAME="$(basename "$0")"
if [ -z "$1" ]; then
  echo Usage: "$SCRIPTNAME <build dir tag>"
  TAG="tmp"
  echo Using tmp as the build dir tag since it was not specified.
else
  TAG="$1" 
fi

ROOT_DIR="$(dirname "$0")"/../
BUILD_DIR_NAME=exp-"$TAG"-rel-gcc-12
BUILD_DIR="$ROOT_DIR""$BUILD_DIR_NAME"
mkdir -p "$BUILD_DIR"
echo Build directory: "$BUILD_DIR_NAME"
echo CMake setup...
cmake -S "$ROOT_DIR" -B "$BUILD_DIR" -GNinja -DBUILD_TEST=ON -DCMAKE_C_COMPILER=gcc-12 -DCMAKE_CXX_COMPILER=g++-12 -DCMAKE_BUILD_TYPE=Release &&
cd "$BUILD_DIR" &&
ninja &&

if [ "$2" == "-s" ]; then
  echo "Setting up the system."
  ../scripts/setup_system.sh
else
  echo "Not execution the system setup. If you want to run the system, add -s as the second option."
fi

# Run benchmarks
./mema-bench
