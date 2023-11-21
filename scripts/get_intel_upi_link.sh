#! /usr/bin/env bash

echo "Downloading Intel PCM..."
git clone https://github.com/intel/pcm.git --branch 202311 --single-branch --recursive
mkdir pcm/build
echo "Building PCM..."
cmake -S ./pcm -B ./pcm/build
cmake --build ./pcm/build --config Release --parallel
echo "Briefly track UPI link traffic to identify the number of links..."
sudo ./pcm/build/bin/pcm -- echo > upi.txt 2>&1
echo "Output written to upi.txt"
