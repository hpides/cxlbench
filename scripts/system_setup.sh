#! /usr/bin/env bash

echo -e "\n### Configure cores to use performance govenor\n" &&
sudo cpupower frequency-set --governor performance &&
# Verify
# cpupower frequency-info -o proc

echo -e "\n### Disabling frequency boosting\n" &&
echo Writing to /sys/devices/system/cpu/cpufreq/boost: &&
echo 0 | sudo tee /sys/devices/system/cpu/cpufreq/boost &&
echo cat /sys/devices/system/cpu/cpufreq/boost: &&
cat /sys/devices/system/cpu/cpufreq/boost &&

echo -e "\n### Disabling SMT\n" &&
echo Writing to /sys/devices/system/cpu/smt/control: &&
echo off | sudo tee /sys/devices/system/cpu/smt/control &&
echo cat /sys/devices/system/cpu/smt/control: &&
cat /sys/devices/system/cpu/smt/control &&

echo -e "\n### Allocating huge pages\n" &&
echo Writing to /proc/sys/vm/nr_hugepages: &&
echo 8000 | sudo tee /proc/sys/vm/nr_hugepages &&
echo cat /proc/sys/vm/nr_hugepages: &&
cat /proc/sys/vm/nr_hugepages

