#!/usr/bin/env python3

import os
import shutil
import subprocess

# Path to the main directory
main_directory = 'vtune'

# Directory where the reduced-tma.txt files will be moved
tma_directory = 'tma'

# Create the tma directory if it does not exist
if not os.path.exists(tma_directory):
    os.makedirs(tma_directory)

# Iterate through all directories in the main directory
for subdir in os.listdir(main_directory):
    subdir_path = os.path.join(main_directory, subdir)

    # # Check if the path is a directory
    if os.path.isdir(subdir_path):
        out_file = f"{os.path.join(tma_directory, subdir)}.csv"
        command = [
            "vtune",
            "-report", "summary",
            "-report-knob", "show-issues=false",
            "-report-output", out_file,
            "-format", "csv",
            "-csv-delimiter", "comma",
            "-r", subdir_path
        ]

        # Run the command
        try:
            subprocess.run(command, check=True)
            print("Report generated successfully.")
        except subprocess.CalledProcessError as e:
            print(f"An error occurred: {e}")
        reduced_tma_path = os.path.join(subdir_path, 'reduced-tma.csv')
