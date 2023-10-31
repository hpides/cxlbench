#! /usr/bin/env bash

ROOT_DIR="$(dirname "$0")"/../
while read -r line
do
  BUILD_DIR_NAME=exp-"$line"-rel-gcc-12
  BUILD_DIR="$ROOT_DIR""$BUILD_DIR_NAME"
  echo "Creating directory: $BUILD_DIR"
  mkdir -p "$BUILD_DIR"
done < "$ROOT_DIR/scripts/exp_labels.txt"

exit 0
