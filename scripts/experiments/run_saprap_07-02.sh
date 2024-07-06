#! /usr/bin/env bash

# ./scripts/run_log.sh saprap ddr_lat_saprap_SNC4 &&
# ./scripts/run_log.sh saprap inter_socket_lat_saprap_SNC4 &&
# ./scripts/run_log.sh saprap cxl_lat_saprap_SNC4 &&
# ./scripts/run_log.sh saprap ddr_bw_saprap_SNC4 &&
# ./scripts/run_log.sh saprap inter_socket_bw_saprap_SNC4 &&
# ./scripts/run_log.sh saprap cxl_bw_saprap_SNC4 &&
# ./scripts/run_log.sh saprap bw_expansion_parallel_saprap_SNC4 &&
./scripts/run_log.sh device_cost_saprap_SNC4 &&
./scripts/run_log.sh fp_tree_saprap_SNC4 &&
# cd exp-saprap-rel-gcc-12/ &&
# mkdir -p results-fs &&
# numactl -N 0,1,2,3 -m 0,1,2,3 ./false-sharing &&
# mv false-sharing.json ./results-fs/false-sharing-N0123-m0123.json &&
# numactl -N 0,1,2,3 -m 4,5,6,7 ./false-sharing &&
# mv false-sharing.json ./results-fs/false-sharing-N0123-m4567.json &&
# numactl -N 0,1,2,3 -m 8 ./false-sharing &&
# mv false-sharing.json ./results-fs/false-sharing-N0123-m8.json
echo "End of experiment series."