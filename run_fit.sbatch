#!/bin/bash
#SBATCH --partition=cpuq
#SBATCH --account=cpuq
#SBATCH --job-name=prospector
#SBATCH --nodes=16
#SBATCH --ntasks=640
#SBATCH --ntasks-per-node=40
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yluo42@ucsc.edu
#SBATCH --output=prospector.out

module load slurm
module load python/3.6.7
module load numpy
module load emcee
module load h5py
module load astropy
module load scipy
module load matplotlib
module load openmpi

export SPS_HOME=/data/groups/leauthaud/yluo42/prospector/lux/fsps
source /data/groups/leauthaud/yluo42/prospector/lux/env/bin/activate

mpirun -n 640 python /data/groups/leauthaud/yluo42/prospector/lux/sed_fitting/pros_fit/fit_prospector_params.py
