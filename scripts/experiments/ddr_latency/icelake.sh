#!/bin/bash
#SBATCH -A epic
#SBATCH --job-name=ddr-lat-icelake
#SBATCH --mail-user=marcel.weisgut@hpi.de
#SBATCH --mail-type=ALL
#SBATCH --partition=alchemy
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
#SBATCH --constraint="n,x,CPU_GEN:ICL,CPU_SKU:8352Y"
#SBATCH --hint=nomultithread
#SBATCH --cpu-freq=Performance

enroot start --rw -r -m /hpi/fs00/home/marcel.weisgut/enroot_mount:/enroot_mount mrclw+dev+x86 bash /enroot_mount/mema-bench/scripts/run_log.sh icelake ddr_lat
