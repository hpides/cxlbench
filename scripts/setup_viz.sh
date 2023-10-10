#!/usr/bin/env bash

DIR="$( cd "$( dirname "$0" )" >/dev/null 2>&1 && pwd )"

if [ ! -d "$DIR/mema_venv" ]; then
    virtualenv "$DIR/mema_venv" &&
    source "$DIR/mema_venv/bin/activate" &&
    pip3 install -r "$DIR/requirements.txt"
else
    source "$DIR/mema_venv/bin/activate"
fi
