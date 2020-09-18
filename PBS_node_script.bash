#!/bin/bash
#PBS -l nodes=1:ppn=16
#PBS -l walltime=48:00:00
#PBS -N MCMC_test

export directory=/nobackup/planetseismology/panning/
export SRC_DIR=/home/panning/git/tremor
export program="$SRC_DIR/MCMC_main.py"
source /home/panning/.bashrc
conda activate obspy
which python
# conda info
export PBS_O_WORKDIR=$directory
cd $PBS_O_WORKDIR

export STARTOFFSET=0
for i in {1..16}; do
    echo "Launching " $[i+$STARTOFFSET]
    echo python $program -d $directory -p $[i+$STARTOFFSET]
    python $program -d $directory -p $[i+$STARTOFFSET] &
    # python $program --version
done
wait
