#! /usr/bin/env bash

./scripts/setup_system.sh &&
./scripts/run_log.sh genoa ddr_lat_genoa_NPS4 &&
./scripts/run_log.sh genoa cxl_lat_genoa_NPS4 &&
cd exp-genoa-rel-gcc-12/ &&
make -j false-sharing &&
mkdir -p results-fs &&
numactl -N 0,1,2,3 -m 0,1 ./false-sharing &&
mv false-sharing.json ./results-fs/false-sharing-N0123-m01.json &&
numactl -N 0,1,2,3 -m 4 ./false-sharing &&
mv false-sharing.json ./results-fs/false-sharing-N0123-m4.json &&
echo "End of experiment series."
