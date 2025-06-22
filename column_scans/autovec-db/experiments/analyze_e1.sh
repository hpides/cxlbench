#!/usr/bin/env bash

# toplev
# toplev --per-socket --nodes '+Frontend_Bound*/2,+Backend_Bound*/2,+Memory_Bound*/3,+Core_Bound*/3,+DRAM_Bound*/4,+MEM_Latency*/5,+MEM_Bandwidth*/5,+MUX' -v --no-desc --core S1 --force-cpu spr ./build/int_scan --benchmark_filter=E1A100_AllLocal100.* --benchmark_repetitions=1

# vtune
# vtune -collect performance-snapshot -r asdf007 ./build/int_scan --benchmark_filter=E1A100_AllLocal100.* --benchmark_repetitions=1
# vtune -report summary -report-knob show-issues=false -report-output tma.txt -r ./r003ps

rm -rf build && mkdir build && cd build && cmake .. && make -j 10 && cd .. &&

placements=(
    # 100 % selectivity
    "E1PA_AllLocalT40S1000"
    "E1PA_AllCXL1BladeT40S1000"
    "E1PA_AllCXL4BladesT40S1000"
    # 0.1 % selectivity
    "E1PA_AllLocalT40S1"
    "E1PA_AllCXL1BladeT40S1"
    "E1PA_AllCXL4BladesT40S1"
)

for placement in "${placements[@]}"; do
    timestamp=$(date +"%Y%m%dT%H%M%S")
    id="${timestamp}_${placement}"
    result_dir="./vtune/${id}"
    reduced_tma="./tma/${id}-reduced.txt"
    export VTUNE_DIR=${result_dir}
    vtune -collect performance-snapshot -no-summary -start-paused -r "${result_dir}" ./build/int_scan --benchmark_filter="${placement}.*" --benchmark_repetitions=1
    vtune -report summary -report-knob show-issues=false -report-output "tma/${id}.txt" -r "${result_dir}"
    vtune -report summary -report-knob show-issues=false -report-output "tma/${id}.csv" -r "${result_dir}" -format csv -csv-delimiter comma
    ./scripts/crop_tma.sh "tma/${id}.txt" "${reduced_tma}"
done
