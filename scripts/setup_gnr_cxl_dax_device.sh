#!/usr/bin/env bash

sudo daxctl reconfigure-device --mode=devdax dax0.0 --force &&
sudo chmod 666 /dev/dax0.0
