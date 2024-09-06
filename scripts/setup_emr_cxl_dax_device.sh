#!/usr/bin/env bash

sudo daxctl reconfigure-device --mode=devdax dax2.0 --force &&
sudo chmod 666 /dev/dax2.0
