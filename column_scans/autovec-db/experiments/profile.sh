#!/usr/bin/env bash

rm -rf build && mkdir build && cd build && cmake .. && make -j 10 && cd .. &&

placements=(
    "E1A_100_AllLocal"
    "E1A_100_ColumnsCXL1Blade"
    "E1A_100_ColumnsCXL4Blades"
    "E1A_100_AllCXL1Blade"
    "E1A_100_AllCXL4Blades"
    "E1A_1_AllLocal"
    "E1A_1_ColumnsCXL1Blade"
    "E1A_1_ColumnsCXL4Blades"
    "E1A_1_AllCXL1Blade"
    "E1A_1_AllCXL4Blades"
)

for placement in "${placements[@]}"; do
    timestamp=$(date +"%Y%m%dT%H%M%S")
    id="${timestamp}_${placement}"
    toplev --core S1 -l1 --force-cpu spr -v --nodes '+Frontend_Bound*/2,+Backend_Bound*/2,+Memory_Bound*/3,+DRAM_Bound*/4,+MEM_Latency*/5,+MEM_Bandwidth*/5,+MUX' -o "${id}-toplev.txt" ./build/int_scan --benchmark_filter="${placement}.*" --benchmark_repetitions=4 --benchmark_out="${id}_result.json" 2>&1 | tee "${id}.log"
    # ./build/int_scan --benchmark_filter="${placement}.*" --benchmark_repetitions=4 --benchmark_out="${placement}_${timestamp}_result.json"
done
