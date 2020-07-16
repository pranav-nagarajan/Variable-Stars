import argparse
import pickle
import multiprocessing as mp
import numpy as np
import pandas as pd
import pymc3 as pm
import theano.tensor as tt

mcmc_parser = argparse.ArgumentParser(description = "Helper for parallel processing.")
mcmc_parser.add_argument('--num_cpus', type = int, help = "Number of processes to use.")
mcmc_parser.add_argument('--calibrate', type = str, help = "Calibration data.")
mcmc_args = mcmc_parser.parse_args()

number_of_cpus = mcmc_args.num_cpus

calibrate = pd.read_csv(mcmc_args.calibrate)
field_periods = np.array(calibrate['Log Period'])
field_moduli = np.array(calibrate['Distance Modulus'])
field_metal = np.array(calibrate['Metallicity'])

observed = np.array(calibrate['Wesenheit Magnitude'])

rr_lyrae_model = pm.Model()

with rr_lyrae_model:

    sigma = pm.HalfNormal('sigma', sd = 1)

    field_zero_point = pm.Normal('calibration_point', mu = 0, sd = 2)
    period_slope = pm.Normal('period_slope', mu = 0, sd = 10)
    metal_slope = pm.Normal('metallicity_slope', mu = 0, sd = 10)

    modeled = []

    for i in range(len(calibrate)):

        modeled.append(field_moduli[i] + field_zero_point + period_slope * field_periods[i]
                       + metal_slope * field_metal[i])

    obs = pm.Normal('obs', mu = modeled, sd = sigma, observed = observed)

with rr_lyrae_model:

    rr_lyrae_trace = pm.sample(cores = number_of_cpus)

pickle.dump(rr_lyrae_trace, open('rr_lyrae_calibration.pkl', 'wb'))
