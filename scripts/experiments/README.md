Genoa
```bash
ssh des-cxl-sg
```
Milan (DELL R7525) | gx[03-05]
```bash
srun -A epic -p sorcery --exclusive --nodes=1 --ntasks=1 --ntasks-per-node=1 --mem=0 --tmp=2G --time=5:00:00 --constraint="g,x,CPU_GEN:MILAN,CPU_SKU:7413" --pty bash
```
Rome (HPE XL225n Gen10) | cx[17-32]
```bash
srun -A epic -p magic --exclusive --nodes=1 --ntasks=1 --ntasks-per-node=1 --mem=0 --tmp=2G --time=5:00:00 --constraint="c,x,CPU_GEN:ROME,CPU_SKU:7742" --pty bash
```
SapRapI
```bash
```
SapRapS
```bash
```
IceLake (DELL R750) | nx[05-06] | For some unknown reason, no resource can be allocated when `--tmp` is used.
```bash
srun -A epic -p alchemy --exclusive --nodes=1 --ntasks=1 --ntasks-per-node=1 --mem=0 --time=5:00:00 --constraint="n,x,CPU_GEN:ICL,CPU_SKU:8352Y" --pty bash
```
CascLake (HPE DL380 Gen10) | nx[03-04]
```bash
srun -A epic -p alchemy --exclusive --nodes=1 --ntasks=1 --ntasks-per-node=1 --mem=0 --time=5:00:00 --constraint="n,x,CPU_GEN:CSL,CPU_SKU:6240L" --pty bash
```
IvyTown (SGI)
```bash
ssh rapa
```
Power9 (IBM IC922) | fp02
```bash
srun -A epic -p magic --nodelist=fp02 --exclusive --mem=0 --time=5:00:00 --cpus-per-task=128 --pty bash
```
`--temp` is also not working here.
```bash
srun -A epic -p magic --exclusive --nodes=1 --ntasks=1 --ntasks-per-node=1 --mem=0 --time=5:00:00 --constraint="f,p,CPU_GEN:POWER9,CPU_SKU:Monza" --pty bash
```

### Running Experiments

Before running an experiment in a container, make sure that MemA is available at `/enroot_mount/mema-bench`. For this, store MemA at `/hpi/fs00/home/marcel.weisgut/enroot_mount/mema-bench` and mount `/hpi/fs00/home/marcel.weisgut/enroot_mount` to the container at `/enroot_mount`.

Example:
```bash
enroot start --rw -r -m /hpi/fs00/home/marcel.weisgut/enroot_mount:/enroot_mount mrclw+dev+x86
```

### Build Directories

Create build directories in `enroot_mount` for each system.

Example:
```
exp-casclake-rel-gcc-12
exp-icelake-rel-gcc-12
exp-lagrange-rel-gcc-12
exp-milan-rel-gcc-12
exp-rome-rel-gcc-12
```
