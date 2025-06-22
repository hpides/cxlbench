#!/usr/bin/env bash

timestamp=$(date +"%Y%m%dT%H%M%S")

rm -rf build && mkdir build && cd build && cmake .. && make -j 10 && cd .. &&
# --benchmark_format=csv
./build/int_scan --benchmark_filter=E2_.* --benchmark_repetitions=4 --benchmark_out="E2_${timestamp}_result.json"
