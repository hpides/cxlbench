#! /usr/bin/env bash

sbatch ./casclake.sh &&
sbatch ./icelake.sh &&
#sbatch ./milan.sh &&
sbatch ./rome.sh &&
#sbatch ./power9.sh &&
exit 0

