#! /usr/bin/env bash

# Exit the script immediately if an error occurs.
set -e

SCRIPTNAME="$(basename "$0")"
if [ -z "$1" ] || [ -z "$2" ] ; then
  echo "Usage: "$SCRIPTNAME" <build dir tag> <workload tag>"
  exit 1
fi

SCRIPT_DIR="$(dirname "$0")"
ROOT_DIR="$(cd "$(dirname "$0")/../" && pwd)"
START_TIME=$(eval date "+%FT%H-%M-%S-%N")
eval "$SCRIPT_DIR/build_run.sh $1 $2" 2>&1 | tee "$ROOT_DIR"/"$1"-"$2"-"$START_TIME".log
