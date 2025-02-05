#!/bin/bash

#SBATCH --job-name=GH200-Throughput
#SBATCH --account rabl
#SBATCH --partition=sorcery
#SBATCH --nodelist=ga01,ga02
#SBATCH --mail-user=marcel.weisgut@hpi.de
#SBATCH --mail-type=ALL
#SBATCH --time=10:00:00
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --container-mounts=/hpi/fs00/home/marcel.weisgut/enroot_mount:/enroot_mount
#SBATCH --container-image=/hpi/fs00/share/fg-rabl/delab-images/arm_ubuntu22_04.sqsh
# Request a minimum and maximum of one node.
#SBATCH --ntasks=1
# Set the maximum of tasks per node to 1. Related to --cpus-per-task but does not require knowledge of the actual
# number of cpus on each node.
# #SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
# Request all the memory on a node
#SBATCH --mem=0
#SBATCH --hint=nomultithread
#SBATCH --cpu-freq=Performance
# #SBATCH --tmp=2G
# #SBATCH --constraint="n,x,CPU_GEN:CSL,CPU_SKU:6240L"

bash touch /enroot_mount/cxlbench/gh200-results/test.txt
