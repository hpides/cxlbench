#!/bin/bash

# Check if the correct number of arguments have been passed
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <server_name> <json_path>"
    exit 1
fi

# Extract the server_name and json_path from the command line arguments
server_name=$1
json_path=$2

# Create the destination directorys if it doesn't exist
mkdir -p ./results/sys/$server_name/plots

# Use scp to copy the json file from the server to the destination directory
scp $server_name:$json_path ./results/sys/$server_name/

# Run the plot script to generate the plots
source ./scripts/setup_viz.sh
./scripts/plot_results.py ./results/sys/$server_name/$(basename $json_path) -o ./results/sys/$server_name/plots/
