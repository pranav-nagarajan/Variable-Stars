import argparse
import pickle
import multiprocessing as mp
import numpy as np
import pandas as pd
import pymc3 as pm

mcmc_parser = argparse.ArgumentParser(description = "Helper for parallel processing.")
mcmc_parser.add_argument('number_of_cpus', metavar = 'N', type = int, help = "Number of processes to use.")
mcmc_parser.add_argument('cepheid_data', type = str, help = "Data for Cepheid stars.")
mcmc_args = mcmc_parser.parse_args()

lin_reg_table = pd.read_csv(mcmc_args.cepheid_data)
number_of_cpus = mcmc_args.number_of_cpus

log_period =  lin_reg_table['Log Period'].values
rel_metal = lin_reg_table['Relative Metallicity'].values
obs_mag = lin_reg_table['Wesenheit Magnitude'].values
galaxy_id = lin_reg_table['Galaxy Code'].values

cepheid_model = pm.Model()

with cepheid_model:

    mod_mu = pm.Normal('mod_mu', mu = 0, sigma = 10)
    mod_sig = pm.HalfNormal('mod_sigma', sigma = 1)

    mod = pm.Normal('mod', mu = mod_mu, sigma = mod_sig, shape = 20)

    zpw = pm.Normal('zpw', mu = 0, sigma = 10)
    bw = pm.Normal('bw', mu = 0, sigma = 10)
    zw = pm.Normal('zw', mu = 0, sigma = 10)

    sigma = pm.HalfNormal('sigma', sigma = 1)

    mag = mod[galaxy_id] + zpw + bw * log_period + zw * rel_metal

    obs = pm.Normal('obs', mu = mag, sigma = sigma, observed = obs_mag)

with cepheid_model:

    cepheid_trace = pm.sample(draws = 500, tune = 1000, cores = number_of_cpus, return_inferencedata = False)

pickle.dump(cepheid_trace, open('cepheid.pkl', 'wb'))
