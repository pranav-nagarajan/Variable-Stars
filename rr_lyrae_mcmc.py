import argparse
import pickle
import multiprocessing as mp
import numpy as np
import pandas as pd
import pymc3 as pm
import theano.tensor as tt

mcmc_parser = argparse.ArgumentParser(description = "Helper for parallel processing.")
mcmc_parser.add_argument('--num_cpus', type = int, help = "Number of processes to use.")
mcmc_parser.add_argument('--data', action = "append", type = str, help = "Data for RR Lyrae stars.")
mcmc_parser.add_argument('--zero_point', nargs = 2, type = float, help = "Theoretical zero point.")
mcmc_parser.add_argument('--metal', nargs = 2, action = "append", type = float, help = "Mean metallicity in galaxy.")
mcmc_args = mcmc_parser.parse_args()

number_of_cpus = mcmc_args.num_cpus

lin_reg_tables = []
for table in mcmc_args.data:
    lin_reg_tables.append(pd.read_csv(table))

metals = mcmc_args.metal
zp, zp_error = mcmc_args.zero_point[0], mcmc_args.zero_point[1]

log_periods = []
obs_mags = []
star_nums = []
star_ids = []

for lin_reg_table in lin_reg_tables:
    log_periods.append(lin_reg_table['Log Period'].values)
    obs_mags.append(lin_reg_table['Wesenheit Magnitude'].values)
    star_nums.append(len(lin_reg_table['Star'].unique()))
    star_ids.append(lin_reg_table['Star Code'].values)

rr_lyrae_model = pm.Model()

with rr_lyrae_model:

    modulus = pm.Normal('modulus', mu = 20, sd = 10)
    sigma = pm.HalfNormal('sigma', sd = 1)

    zero_point = pm.Normal('zero_point', mu = zp, sd = zp_error)
    period_slope = pm.Normal('period_slope', mu = 0, sd = 10)
    metal_slope = pm.Normal('metallicity_slope', mu = 0, sd = 10)

    magnitudes = []

    for i in range(len(log_periods)):

        metal = pm.Normal(f'metallicity_{i}', mu = metals[i][0], sd = metals[i][1], shape = star_nums[i])
        magnitudes.append(modulus + zero_point + period_slope * log_periods[i] + metal_slope * metal[star_ids[i]])

    modeled, observed = pm.math.concatenate(magnitudes), pm.math.concatenate(obs_mags)
    obs = pm.Normal('obs', mu = modeled, sd = sigma, observed = observed)

with rr_lyrae_model:

    rr_lyrae_trace = pm.sample(cores = number_of_cpus)

pickle.dump(rr_lyrae_trace, open('rr_lyrae.pkl', 'wb'))
