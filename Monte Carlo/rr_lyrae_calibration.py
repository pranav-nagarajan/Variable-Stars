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

obs_mags = []
obs_mags.append(calibrate['B-V Wesenheit Magnitude'].values)
obs_mags.append(calibrate['V-I Wesenheit Magnitude'].values)

errors = []
field_mag_err_BV = calibrate['Uncertainty in B-V Wesenheit Magnitude'].values
field_mag_err_VI = calibrate['Uncertainty in V-I Wesenheit Magnitude'].values
field_mod_err = calibrate['Uncertainty in Distance Modulus'].values
errors.append(np.sqrt(field_mag_err_BV**2 + field_mod_err**2))
errors.append(np.sqrt(field_mag_err_VI**2 + field_mod_err**2))
errors = np.hstack(errors)

rr_lyrae_model = pm.Model()

with rr_lyrae_model:

    sigma_BV = pm.HalfNormal('sigma_BV', sd = 1)
    sigma_VI = pm.HalfNormal('sigma_VI', sd = 1)

    zero_point_BV = pm.Normal('calibration_point_BV', mu = 0, sd = 10)
    period_slope_BV = pm.Normal('period_slope_BV', mu = 0, sd = 10)
    metal_slope_BV = pm.Normal('metallicity_slope_BV', mu = 0, sd = 10)

    zero_point_VI = pm.Normal('calibration_point_VI', mu = 0, sd = 10)
    period_slope_VI = pm.Normal('period_slope_VI', mu = 0, sd = 10)
    metal_slope_VI = pm.Normal('metallicity_slope_VI', mu = 0, sd = 10)

    modeled_BV = (zero_point_BV + field_moduli + period_slope_BV * (field_periods + 0.3)
                  + metal_slope_BV * (field_metal + 1.36))

    modeled_VI = (zero_point_VI + field_moduli + period_slope_VI * (field_periods + 0.3)
                  + metal_slope_VI * (field_metal + 1.36))

    sigmas = []
    for i in range(len(field_periods)):
        sigmas.append(sigma_BV)
    for i in range(len(field_periods)):
        sigmas.append(sigma_VI)

    sigmas = tt.as_tensor_variable(sigmas)
    total_err = np.sqrt(sigmas**2 + errors**2)

    magnitudes = []
    magnitudes.append(modeled_BV)
    magnitudes.append(modeled_VI)

    modeled, observed = pm.math.concatenate(magnitudes), pm.math.concatenate(obs_mags)

    obs = pm.Normal('obs', mu = modeled, sd = total_err, observed = observed)

with rr_lyrae_model:

    rr_lyrae_trace = pm.sample(cores = number_of_cpus)

pickle.dump(rr_lyrae_trace, open('rr_lyrae_calibration.pkl', 'wb'))
