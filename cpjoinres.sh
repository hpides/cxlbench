#!/usr/bin/env bash

# Define the source path as a variable
source_path="~/cxlbench/third_party/in-memory-hash-joins/experiments"

# Check if an argument was passed to the script
if [ -z "$1" ]; then
    # If no argument provided, show option information
    echo "Choose the source machine:"
    echo "m: mag-SYS-741GE-TNRT"
    echo "1: des-node01"
    echo "2: des-node02"
    echo "3: des-node03"
    read -p "Enter your choice (m/1/2/3): " choice
else
    # If argument provided, use it as the choice
    choice=$1
fi

# Define the destination folder with suffix based on the selected option
destination="./joinres-$choice"

# React to user's choice
case $choice in
  m)
    scp -r mag-SYS-741GE-TNRT:$source_path $destination
    ;;
  1)
    scp -r des-node01:$source_path $destination
    ;;
  2)
    scp -r des-node02:$source_path $destination
    ;;
  3)
    scp -r des-node03:$source_path $destination
    ;;
  *)
    echo "Invalid choice. Please enter m, 1, 2, or 3."
    ;;
esac

