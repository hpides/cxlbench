#! /usr/bin/env bash

./scripts/setup_system.sh &&
# ./scripts/run_log.sh ddr_bw &&
# ./scripts/run_log.sh inter_socket_bw &&
# ./scripts/run_log.sh cxl_bw_emr_SNC1 &&
#./scripts/run_log.sh ddr_lat &&
#./scripts/run_log.sh inter_socket_lat &&
#./scripts/run_log.sh cxl_lat_emr_SNC1 &&
#./scripts/run_log.sh bw_expansion_parallel_emr_SNC1 &&
# ./scripts/run_log.sh bw_expansion_parallel_emr_SNC1_4k &&
# ./scripts/run_log.sh device_cost_emr_SNC1 &&
#./scripts/run_log.sh fp_tree_emr_SNC1 &&
./scripts/run_log.sh cxl_bw_emr_1-4cxl &&
#cd exp-mag-SYS-741GE-TNRT-rel-gcc-12/ &&
#mkdir -p results-fs &&
#numactl -N 0 -m 0 ./false-sharing &&
#mv false-sharing.json ./results-fs/false-sharing-N0-m0.json &&
#numactl -N 0 -m 1 ./false-sharing &&
#mv false-sharing.json ./results-fs/false-sharing-N0-m1.json &&
#numactl -N 0 -m 2 ./false-sharing &&
#mv false-sharing.json ./results-fs/false-sharing-N0-m2.json &&
echo "End of experiment series."
