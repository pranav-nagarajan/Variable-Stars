import argparse
import pickle
import multiprocessing as mp
import numpy as np
import pandas as pd
import pymc3 as pm
import theano.tensor as tt

mcmc_parser = argparse.ArgumentParser(description = "Helper for parallel processing.")
mcmc_parser.add_argument('number_of_cpus', metavar = 'N', type = int, help = "Number of processes to use.")
mcmc_parser.add_argument('cepheid_data', type = str, help = "Data for Cepheid stars.")
mcmc_args = mcmc_parser.parse_args()

lin_reg_4258 = pd.read_csv(mcmc_args.cepheid_data)
number_of_cpus = mcmc_args.number_of_cpus

# log_period =  lin_reg_table['Log Period'].values
# rel_metal = lin_reg_table['Relative Metallicity'].values
# obs_mag = lin_reg_table['Wesenheit Magnitude'].values
# galaxy_id = lin_reg_table['Galaxy Code'].values
#
# num_gals = len(lin_reg_table['Galaxy'].unique())
# mask = np.array(lin_reg_table['Galaxy'] == 'N4258')
#
# cepheid_model = pm.Model()
#
# with cepheid_model:
#
#     mod = pm.Uniform('mod', lower = -10, upper = 10, shape = (num_gals - 1))
#     rel_mod = tt.set_subtensor(mod[galaxy_id - 1][mask.nonzero()], 0)
#
#     zpw = pm.Uniform('zpw', lower = 20, upper = 30)
#     bw = pm.Uniform('bw', lower = -10, upper = 10)
#     zw = pm.Uniform('zw', lower = -10, upper = 10)
#
#     err = lin_reg_table['Uncertainty in Magnitude (F160W)'].values
#     var = pm.HalfNormal('var', sd = 1)
#     sigma = np.sqrt(err**2 + var)
#
#     mag = rel_mod + zpw + bw * log_period + zw * rel_metal
#
#     obs = pm.Normal('obs', mu = mag, sd = sigma, observed = obs_mag)
#
# with cepheid_model:
#     cepheid_trace = pm.sample(cores = number_of_cpus, return_inferencedata = False)
#
# pickle.dump(cepheid_trace, open('cepheid.pkl', 'wb'))

NGC_model = pm.Model()

with NGC_model:

    zpw = pm.Normal('zpw', mu = 0, sd = 10)
    bw = pm.Normal('bw', mu = 0, sd = 10)
    zw = pm.Normal('zw', mu = 0, sd = 10)

    err = lin_reg_4258['Uncertainty in Magnitude (F160W)'].values
    var = pm.HalfNormal('var', sd = 1)
    sigma = np.sqrt(err**2 + var)

    mag = zpw + bw * lin_reg_4258['Log Period'].values + zw * lin_reg_4258['Relative Metallicity'].values

    obs = pm.Normal('obs', mu = mag, sd = sigma, observed = lin_reg_4258['Wesenheit Magnitude'].values)

with NGC_model:
    NGC_trace = pm.sample(draws = 500, tune = 1000)

pickle.dump(NGC_trace, open('NGC_4258.pkl', 'wb'))
