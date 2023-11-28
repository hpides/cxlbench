#!/bin/bash

# Check if the correct number of arguments have been passed
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <server_name> <remote_result_dir> <local_result_dir"
    exit 1
fi

# Extract the server_name and json_path from the command line arguments
server_name=$1
remote_result_dir=$2
local_result_dir=$3

mkdir -p $local_result_dir
scp -r $server_name:$remote_result_dir $local_result_dir
