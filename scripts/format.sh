#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
find "$DIR/../src" -iname '*.hpp' -o -iname '*.cpp' | xargs clang-format -i -style=file
find "$DIR/../test" -iname '*.hpp' -o -iname '*.cpp' | xargs clang-format -i -style=file

# Python formatting
cd "$DIR" || exit 1 # cd scripts
poetry run ruff format .

