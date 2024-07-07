#!/bin/bash
#SBATCH -A epic
#SBATCH --job-name=general-power9
#SBATCH --mail-user=marcel.weisgut@hpi.de
#SBATCH --mail-type=ALL
#SBATCH --partition=magic
#SBATCH --time=10:00:00
#SBATCH --exclusive
#SBATCH --nodes=1
# Request a minimum and maximum of one node.
#SBATCH --ntasks=1
# Set the maximum of tasks per node to 1. Related to --cpus-per-task but does not require knowledge of the actual
# number of cpus on each node.
#SBATCH --ntasks-per-node=1
# Request all the memory on a node
#SBATCH --mem=0
# Not working
##SBATCH --tmp=2G
#SBATCH --constraint="f,p,CPU_GEN:POWER9,CPU_SKU:Monza"
#SBATCH --hint=nomultithread
#SBATCH --cpu-freq=Performance

SCRIPTNAME="$(basename "$0")"
if [ -z "$1" ]; then
  echo "Usage: "$SCRIPTNAME" <workload tag>"
  exit 1
fi

enroot import docker://mrclw/dev:ppc64le
enroot create mrclw+dev+ppc64le.sqsh
enroot start --rw -r -m /hpi/fs00/home/marcel.weisgut/enroot_mount:/enroot_mount mrclw+dev+ppc64le bash /enroot_mount/mema-bench/scripts/run_log.sh power9 "$1"
