#! /usr/bin/env bash

# Exit the script immediately if an error occurs.
set -e

SCRIPTNAME="$(basename "$0")"
if [ -z "$1" ] || [ -z "$2" ] ; then
  echo "Usage: "$SCRIPTNAME" <build dir tag> <workload tag>"
  exit 1
fi

TAG="$1"
WORKLOAD="$2"
START_TIME=$(eval date "+%FT%H-%M-%S-%N")

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RESULT_DIR="$ROOT_DIR"/results/"$WORKLOAD"/"$TAG"/"$START_TIME"
mkdir -p "$RESULT_DIR"

SCRIPT_DIR="$(dirname "$0")"
eval "$SCRIPT_DIR/run.sh $1 $2 $START_TIME" 2>&1 | tee "$RESULT_DIR"/all-steps-"$START_TIME".log
