#! /usr/bin/env bash

# This script might not work on all machines. Feel free to open an issue mentioning the system and error message that
# you receive when executing the script.
echo -e "\n### Perfoemance govenor\n"
cpupower frequency-info

echo -e "\n### Frequency boosting\n"
echo cat /sys/devices/system/cpu/cpufreq/boost:
cat /sys/devices/system/cpu/cpufreq/boost

echo -e "\n### SMT\n"
echo cat /sys/devices/system/cpu/smt/control:
cat /sys/devices/system/cpu/smt/control

echo -e "\n### Allocated huge pages\n"
echo cat /proc/sys/vm/nr_hugepages:
cat /proc/sys/vm/nr_hugepages
