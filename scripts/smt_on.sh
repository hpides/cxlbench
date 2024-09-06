#! /usr/bin/env bash

echo -e "\n### Enable SMT\n" &&
echo Writing to /sys/devices/system/cpu/smt/control: &&
echo on | sudo tee /sys/devices/system/cpu/smt/control &&
echo cat /sys/devices/system/cpu/smt/control: &&
cat /sys/devices/system/cpu/smt/control &&

echo Done.
