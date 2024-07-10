#! /usr/bin/env bash

sudo apt update &&
sudo apt install -y \
    cmake \
    dmidecode \
    g++-12 \
    gcc-12 \
    gdb \
    git \
    hwloc \
    libnuma-dev \
    lshw \
    ninja-build \
    numactl \
    pciutils \
    ruby \
    vim \
    zlib1g-dev \
    pipx

pipx install poetry  # python package manager
pipx ensurepath
echo "To install python dependencies run 'poetry install'"

