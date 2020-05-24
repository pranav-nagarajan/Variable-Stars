#!/bin/bash
#SBATCH --job-name=hubble
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
echo "Starting period computation for Hubble RR Lyrae dataset..."
python compute_period.py $SLURM_CPUS_PER_TASK
