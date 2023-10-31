#! /usr/bin/env bash

# This scripts copies the yaml files found in the passed path to all `exp-` directories present in the root directory.
SCRIPTNAME="$(basename "$0")"
if [ -z "$1" ]; then
  echo "Usage: $SCRIPTNAME <path to workload configuration files>"
  exit 1 
fi

if [ ! -d "$1" ]; then
  echo "Passed argument $1 is not a directory."
  exit 1 
fi

ROOT_DIR="$(dirname "$0")"/../
for expdir in "$ROOT_DIR"/exp-* ;
do
  target_dir="$expdir"/workloads
  mkdir -p "$target_dir"
  for yamlfile in "$1"/*.yaml ; 
  do
    target_filename="$(basename "$yamlfile")"
    cp "$yamlfile" "$target_dir"/"$target_filename"
  done;
done;

exit 0
