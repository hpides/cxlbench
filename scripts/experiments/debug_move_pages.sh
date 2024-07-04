#! /usr/bin/env bash

rm -f move_pages_debug.txt &&
numactl -H | tee -a move_pages_debug.txt &&
# cat /boot/config-$(uname -r) | tee -a move_pages_debug.txt &&
# ulimit -a | tee -a move_pages_debug.txt &&
mkdir -p build_move_pages &&
cd build_move_pages &&
cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_C_COMPILER=gcc-12 -DCMAKE_CXX_COMPILER=g++-12 &&
make -j 20 check_move_pages &&
sudo setcap 'cap_sys_nice=eip' ./check_move_pages &&
./check_move_pages 0 1 1024 | tee -a ../move_pages_debug.txt &&
./check_move_pages 1 2 1024 | tee -a ../move_pages_debug.txt &&
./check_move_pages 2 3 1024 | tee -a ../move_pages_debug.txt &&
./check_move_pages 3 4 1024 | tee -a ../move_pages_debug.txt &&
./check_move_pages 4 5 1024 | tee -a ../move_pages_debug.txt &&
./check_move_pages 5 6 1024 | tee -a ../move_pages_debug.txt &&
./check_move_pages 6 7 1024 | tee -a ../move_pages_debug.txt &&
./check_move_pages 7 8 1024 | tee -a ../move_pages_debug.txt &&
./check_move_pages 8 7 1024 | tee -a ../move_pages_debug.txt &&
cd ..
# cd .. &&
# ./scripts/run_log.sh saprap ddr_lat_saprap_SNC4 2>&1 | tee -a move_pages_debug.txt
