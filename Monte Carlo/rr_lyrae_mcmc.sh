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
python rr_lyrae_mcmc.py --num_cpus $SLURM_CPUS_PER_TASK --data and_one_mcmc.csv --data and_two_mcmc.csv --data and_three_mcmc.csv --data and_eleven_mcmc.csv --data and_thirteen_mcmc.csv  --data and_fifteen_mcmc.csv --data and_sixteen_mcmc.csv --data and_twenty_eight_mcmc.csv --data bootes_mcmc.csv --data carina_mcmc.csv --data carina_two_mcmc.csv --data cetus_mcmc.csv --data coma_mcmc.csv --data crater_mcmc.csv --data draco_mcmc.csv --data eso_294_mcmc.csv --data eso_410_mcmc.csv --data fornax_mcmc.csv --data grus_mcmc.csv --data ic_mcmc.csv --data leo_a_mcmc.csv --data leo_i_mcmc.csv --data leo_p_mcmc.csv --data leo_t_mcmc.csv --data NGC_147_mcmc.csv --data NGC_185_mcmc.csv --data phoenix_mcmc.csv --data phoenix_two_mcmc.csv --data sculptor_mcmc.csv --data tucana_mcmc.csv --galaxies galaxies.csv --calibrate milky_way_mcmc.csv
