#! /usr/bin/env bash

# Controls use of the performance events system by unprivileged users (without CAP_SYS_ADMIN). The default value is 2.
sudo sysctl -w kernel.perf_event_paranoid=0 &&

# Indicates whether restrictions are placed on exposing kernel addresses via /proc and other interfaces.
echo 0 | sudo tee /proc/sys/kernel/kptr_restrict
