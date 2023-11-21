#! /usr/bin/env bash

DIR_NAME=AMDuProf_Linux_x64_4.1.424
FILE_NAME="$DIR_NAME".tar.bz2

echo "Downloading AMD uprof..."
curl https://download.amd.com/developer/eula/uprof/"$FILE_NAME" --output "$FILE_NAME"
echo "Unpacking AMD uprof..."
tar -xf "$FILE_NAME"
echo "Briefly track xGMI link traffic to identify the number of xGMI links..."
sudo ./"$DIR_NAME"/bin/AMDuProfPcm -m xgmi -d 1 > xgmi.txt
echo "Output written to xgmi.txt"
