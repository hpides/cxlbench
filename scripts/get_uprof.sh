#! /usr/bin/env bash

VERSION_NAME=AMDuProf_Linux_x64_4.1.424
FILE_NAME="$VERSION_NAME".tar.bz2

echo "Downloading AMD uprof..."
curl https://download.amd.com/developer/eula/uprof/"$FILE_NAME" --output "$FILE_NAME"
echo "Unpacking AMD uprof..."
tar -xf "$FILE_NAME"
