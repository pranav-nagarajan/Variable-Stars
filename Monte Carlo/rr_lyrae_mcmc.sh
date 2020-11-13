#!/bin/bash
#SBATCH --job-name=mcmc
#SBATCH --account=co_dweisz
#SBATCH --partition=savio2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=02:00:00
#SBATCH --output=test_job_%j.out
#SBATCH --error=test_job_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pranav.njn@berkeley.edu

## Command(s) to run:
module load python/3.6
echo "Starting MCMC for RR Lyrae P-L Relation..."
python rr_lyrae_mcmc.py --num_cpus $SLURM_CPUS_PER_TASK --data and_one_mcmc.csv --data and_two_mcmc.csv --data and_three_mcmc.csv --data and_fifteen_mcmc.csv --data and_twenty_eight_mcmc.csv --data cetus_mcmc.csv --data crater_mcmc.csv --data ic_mcmc.csv --data leo_a_mcmc.csv --data leo_i_mcmc.csv --data sculptor_mcmc.csv --data tucana_mcmc.csv --galaxies galaxies.csv --calibrate milky_way_mcmc.csv
