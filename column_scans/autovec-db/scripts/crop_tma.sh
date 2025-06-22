#!/bin/bash

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 input_file output_file"
  exit 1
fi

input_file=$1
output_file=$2

sed -n '/Top-down Microarchitecture Analysis (TMA)/, /Vectorization:/ { /Vectorization:/q; p }' "$input_file" > "$output_file"

echo "Content cropped and saved to $output_file"
