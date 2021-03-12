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
python rr_lyrae_mcmc.py --num_cpus $SLURM_CPUS_PER_TASK --data canes_mcmc.csv --data canes_two_mcmc.csv --data hercules_mcmc.csv --data leo_iv_mcmc.csv --data sagittarius_two_mcmc.csv --data segue_two_mcmc.csv --data ursa_major_mcmc.csv --data ursa_major_two_mcmc.csv --galaxies galaxies_BV.csv --calibrate milky_way_BV.csv
