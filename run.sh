#!/bin/bash
#PBS -l nodes=4:ppn=4
#PBS -l walltime=2:00:00
#PBS -q pace-ice
#PBS -N asingh_amptorch_job
#PBS -o stdout
#PBS -e stderr
cd $PBS_O_WORKDIR

python train_final.py
