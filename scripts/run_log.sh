#! /usr/bin/env bash

# Exit the script immediately if an error occurs.
set -e

SCRIPTNAME="$(basename "$0")"
if [ -z "$1" ]; then
  echo "Usage: "$SCRIPTNAME" [build dir tag] <workload tag>"
  TAG="$(hostname)"
  WORKLOAD="Undefined"
elif [ -z "$2" ]; then
  TAG="$(hostname)"
  WORKLOAD="$1"
elif [ -z "$3" ]; then
  TAG="$1"
  WORKLOAD="$2"
fi

START_TIME=$(eval date "+%FT%H-%M-%S-%N")

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RESULT_DIR="$ROOT_DIR"/results/"$WORKLOAD"/"$TAG"/"$START_TIME"
mkdir -p "$RESULT_DIR"

SCRIPT_DIR="$(dirname "$0")"
eval "$SCRIPT_DIR/run.sh $TAG $WORKLOAD $START_TIME" 2>&1 | tee "$RESULT_DIR"/all-steps-"$START_TIME".log
