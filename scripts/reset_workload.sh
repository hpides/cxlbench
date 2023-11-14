#! /usr/bin/env bash

# This scripts creates a "workloads" script in the current working directory and copies the workload set associated with
# the passed tag to that workloads directory.
SCRIPTNAME="$(basename "$0")"
if [ -z "$1" ]; then
  echo "Error: Usage: $SCRIPTNAME <directory to be copied>"
  echo "Options you can use:"
  for dir in ../workloads/*; do basename "$dir"; done
  exit 1 
fi

# Check if the script was executed in a build directory, which is expected to be a sub-directory of the root directory.
# Using a directory and file that is expected to be located in the projects root directlry.
if [ ! -d "../scripts" ] || [ ! -f "../.clang-format" ] ; then
  echo "Error: Make sure that you execute the script in a build dir."
  exit 1 
fi

if [ ! -d ../workloads/"$1" ]; then
  echo "Error: Make sure that the given name of the subdir exists in the source workloads dir."
  echo "Options you can use:"
  for dir in ../workloads/*; do basename "$dir"; done
  exit 1 
fi

mkdir -p workloads
rm -f ./workloads/*.yaml
rm -f ./workloads/*/*.yaml

cp -r ../workloads/"$1" ./workloads
exit 0
