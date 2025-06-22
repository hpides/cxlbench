#!/usr/bin/env python3

import argparse
import os
import shutil

parser = argparse.ArgumentParser(description='Move reduced-tma.txt files to a specified directory.')
parser.add_argument('vtune_dir', type=str, help='Path to the main directory')
parser.add_argument('tma_dir', type=str, help='Directory where the reduced-tma.txt files will be moved')

# Parse the arguments
args = parser.parse_args()

vtune_dir = args.vtune_dir
tma_dir = args.tma_dir

# Create the tma directory if it does not exist
if not os.path.exists(tma_dir):
    os.makedirs(tma_dir)

# Iterate through all directories in the main directory
for subdir in os.listdir(vtune_dir):
    subdir_path = os.path.join(vtune_dir, subdir)

    # Check if the path is a directory
    if os.path.isdir(subdir_path):
        for filename in os.listdir(subdir_path):
            file_path = os.path.join(subdir_path, filename)
            parts = filename.rsplit(".", 1)
            if parts[-1] not in ["txt", "csv"]:
                continue
            new_filename = f"{subdir}.{parts[-1]}"
            destination_path = os.path.join(tma_dir, new_filename)
            shutil.move(file_path, destination_path)
            print(f"Moved: {file_path} to {destination_path}")

print("Done.")
