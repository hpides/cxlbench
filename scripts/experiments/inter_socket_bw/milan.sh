#!/bin/bash
#SBATCH -A epic
#SBATCH --job-name=inter-bw-milan
#SBATCH --mail-user=marcel.weisgut@hpi.de
#SBATCH --mail-type=ALL
#SBATCH --partition=sorcery
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
#SBATCH --tmp=2G
#SBATCH --constraint="g,x,CPU_GEN:MILAN,CPU_SKU:7413"
#SBATCH --hint=nomultithread
#SBATCH --cpu-freq=Performance

enroot start --rw -r -m /hpi/fs00/home/marcel.weisgut/enroot_mount:/enroot_mount mrclw+dev+x86 bash /enroot_mount/cxlbench/scripts/run_log.sh milan inter_socket_bw_milan
