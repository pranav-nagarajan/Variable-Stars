#!/bin/bash
#SBATCH --job-name=mcmc
#SBATCH --account=co_dweisz
#SBATCH --partition=savio2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=01:00:00
#SBATCH --output=test_job_%j.out
#SBATCH --error=test_job_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pranav.njn@berkeley.edu

## Command(s) to run:
module load python/3.6
echo "Starting MCMC for RR Lyrae P-L Relation..."
python rr_lyrae_mcmc.py --num_cpus $SLURM_CPUS_PER_TASK --data sculptor_mcmc.csv --data and_one_mcmc.csv --data and_two_mcmc.csv --data and_three_mcmc.csv --data and_fifteen_mcmc.csv --data and_twenty_eight_mcmc.csv --data crater_mcmc.csv --metal -1.90 1.0 --metal -1.45 1.0 --metal -1.64 1.0 --metal -1.81 1.0 --metal -1.80 1.0 --metal -1.84 1.0 --metal -1.92 1.0 --calibrate milky_way_mcmc.csv
