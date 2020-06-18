import argparse
import pickle
import multiprocessing as mp
import numpy as np
import pandas as pd
import pymc3 as pm
import theano.tensor as tt

mcmc_parser = argparse.ArgumentParser(description = "Helper for parallel processing.")
mcmc_parser.add_argument('number_of_cpus', metavar = 'N', type = int, help = "Number of processes to use.")
mcmc_parser.add_argument('rr_lyrae_data', type = str, help = "Data for RR Lyrae stars in a specific galaxy.")
mcmc_parser.add_argument('zero_point', type = float, help = "Theoretical zero point.")
mcmc_parser.add_argument('zp_error', type = float, help = "Uncertainty in theoretical zero point.")
mcmc_parser.add_argument('mean_metallicity', type = float, help = "Mean metallicity in galaxy.")
mcmc_parser.add_argument('sd_metallicity', type = float, help = "Standard deviation of metallicity in galaxy.")
mcmc_args = mcmc_parser.parse_args()

lin_reg_table = pd.read_csv(mcmc_args.rr_lyrae_data)
number_of_cpus = mcmc_args.number_of_cpus

mean_metal, sd_metal = mcmc_args.mean_metallicity, mcmc_args.sd_metallicity
zp, zp_error = mcmc_args.zero_point, mcmc_args.zp_error

log_period =  lin_reg_table['Log Period'].values
obs_mag = lin_reg_table['Wesenheit Magnitude'].values

num_stars = len(lin_reg_table['Star'].unique())
star_id = lin_reg_table['Star Code'].values

rr_lyrae_model = pm.Model()

with rr_lyrae_model:

    modulus = pm.Normal('modulus', mu = 0, sd = 10)

    zero_point = pm.Normal('zero_point', mu = zp, sd = zp_error)
    period_slope = pm.Normal('period_slope', mu = 0, sd = 10)
    metal_slope = pm.Normal('metallicity_slope', mu = 0, sd = 10)

    metallicity = pm.Normal('metallicity', mu = mean_metal, sd = sd_metal, shape = num_stars)

    sigma = pm.HalfNormal('sigma', sd = 1)

    mag = modulus + zero_point + period_slope * log_period + metal_slope * metallicity[star_id]

    obs = pm.Normal('obs', mu = mag, sd = sigma, observed = obs_mag)

with rr_lyrae_model:
    rr_lyrae_trace = pm.sample(cores = number_of_cpus)

pickle.dump(rr_lyrae_trace, open('rr_lyrae.pkl', 'wb'))
