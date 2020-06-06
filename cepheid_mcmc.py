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

cepheid_model = pm.Model()

with cepheid_model:

    mod = pm.Normal('mod', mu = 0, sigma = 10)
    zpw = pm.Normal('zpw', mu = 0, sigma = 10)
    bw = pm.Normal('bw', mu = 0, sigma = 10)
    zw = pm.Normal('zw', mu = 0, sigma = 10)
    sigma = pm.HalfNormal('sigma', sigma = 1)

    mag = mod + zpw + bw * lin_reg_table['Log Period'].values + zw * lin_reg_table['Relative Metallicity'].values

    obs = pm.Normal('obs', mu = mag, sigma = sigma, observed = lin_reg_table['Wesenheit Magnitude'].values)

with cepheid_model:

    cepheid_trace = pm.sample(draws = 500, tune = 1000, cores = number_of_cpus)

pickle.dump(cepheid_trace, open('cepheid.pkl', 'wb'))
