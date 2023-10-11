#!/bin/bash

# Check if the correct number of arguments have been passed
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <server_name> <json_path>"
    exit 1
fi

# Extract the server_name and json_path from the command line arguments
server_name=$1
json_path=$2


# Use scp to copy the json file from the server to the destination directory
scp $server_name:$json_path ./results/sys/$server_name/

# Create the destination directorys if it doesn't exist
file_name=$(basename $json_path)
result_id=${file_name%.*}
plot_path=./results/sys/$server_name/plots/$result_id
mkdir -p $plot_path

# Run the plot script to generate the plots
source ./scripts/setup_viz.sh
./scripts/plot_results.py ./results/sys/$server_name/$file_name -o $plot_path

