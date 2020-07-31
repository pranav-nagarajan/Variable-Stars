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
field_periods = calibrate['Log Period'].values
field_moduli = calibrate['Distance Modulus'].values
field_metal = calibrate['Metallicity'].values

observed = calibrate['Wesenheit Magnitude'].values

field_mag_err = calibrate['Uncertainty in Wesenheit Magnitude'].values
field_mod_err = calibrate['Uncertainty in Distance Modulus'].values

rr_lyrae_model = pm.Model()

with rr_lyrae_model:

    sigma = pm.HalfNormal('sigma', sd = 1)
    total_err = np.sqrt(sigma**2 + field_mag_err**2 + field_mod_err**2)

    zero_point = pm.Normal('calibration_point', mu = 0, sd = 10)
    period_slope = pm.Normal('period_slope', mu = 0, sd = 10)
    metal_slope = pm.Normal('metallicity_slope', mu = 0, sd = 10)

    modeled = (zero_point + field_moduli + period_slope * (field_periods + 0.3)
               + metal_slope * (field_metal + 1.36))

    obs = pm.Normal('obs', mu = modeled, sd = total_err, observed = observed)

with rr_lyrae_model:

    rr_lyrae_trace = pm.sample(cores = number_of_cpus)

pickle.dump(rr_lyrae_trace, open('rr_lyrae_calibration.pkl', 'wb'))
