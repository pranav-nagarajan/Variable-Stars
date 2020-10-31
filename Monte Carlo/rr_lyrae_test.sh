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
python rr_lyrae_mcmc.py --num_cpus $SLURM_CPUS_PER_TASK --data galaxy_zero.csv --data galaxy_one.csv --data galaxy_two.csv --metal -2.386 0.50 --metal -1.613 0.50 --metal -2.018 0.50 --calibrate mock_calib_lin_reg.csv
