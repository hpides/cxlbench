#! /usr/bin/env bash

# Exit the script immediately if an error occurs.
set -e

SCRIPTNAME="$(basename "$0")"
if [ -z "$1" ]; then
  echo "Usage: "$SCRIPTNAME" <path to result directory>"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")/" && pwd)"

source "$SCRIPT_DIR"/setup_viz.sh
"$SCRIPT_DIR"/plot_results.py $1
