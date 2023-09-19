#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

if [ ! -d "$DIR/venv" ]; then
    set -e
    virtualenv "$DIR/venv"
    source "$DIR/venv/bin/activate"
    pip3 install -r "$DIR/requirements.txt"
else
    source "$DIR/venv/bin/activate"
fi
