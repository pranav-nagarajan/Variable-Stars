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
mcmc_parser.add_argument('--galaxies', type = str, help = "Catalog of Galaxies.")
mcmc_parser.add_argument('--calibrate', type = str, help = "Calibration data.")
mcmc_args = mcmc_parser.parse_args()

number_of_cpus = mcmc_args.num_cpus

lin_reg_tables = []
for table in mcmc_args.data:
    lin_reg_tables.append(pd.read_csv(table))

galaxies = pd.read_csv(mcmc_args.galaxies)
galaxy_metal = galaxies['Mean Metallicity'].values

log_periods = []
obs_mags = []
star_nums = []
star_ids = []
errors = []

for lin_reg_table in lin_reg_tables:
    log_periods.append(lin_reg_table['Log Period'].values)
    obs_mags.append(lin_reg_table['Wesenheit Magnitude'].values)
    star_nums.append(len(lin_reg_table['Star'].unique()))
    star_ids.append(lin_reg_table['Star Code'].values)
    errors.append(lin_reg_table['Uncertainty in Wesenheit Magnitude'].values)

calibrate = pd.read_csv(mcmc_args.calibrate)
field_periods = calibrate['Log Period'].values
field_moduli = calibrate['Distance Modulus'].values
field_metal = calibrate['Metallicity'].values

field_mags = calibrate['Wesenheit Magnitude'].values
obs_mags.append(field_mags)

field_mag_err = calibrate['Uncertainty in Wesenheit Magnitude'].values
field_mod_err = calibrate['Uncertainty in Distance Modulus'].values
errors.append(np.sqrt(field_mag_err**2 + field_mod_err**2))
errors = np.hstack(errors)

rr_lyrae_model = pm.Model()

with rr_lyrae_model:

    sigma = pm.HalfNormal('sigma', sd = 0.5)

    modulus = pm.Normal('modulus', mu = 20, sd = 10, shape = len(lin_reg_tables))

    # zero_point = pm.Normal('zero_point', mu = -0.94, sd = 0.001)
    # period_slope = pm.Normal('period_slope', mu = -2.43, sd = 0.001)
    # metal_slope = pm.Normal('metallicity_slope', mu = 0.15, sd = 0.001)

    # zero_point = pm.Normal('zero_point', mu = -1.11, sd = 0.001)
    # period_slope = pm.Normal('period_slope', mu = -2.67, sd = 0.001)
    # metal_slope = pm.Normal('metallicity_slope', mu = -0.02, sd = 0.001)

    zero_point = pm.Normal('zero_point', mu = 0, sd = 1)
    period_slope = pm.Normal('period_slope', mu = 0, sd = 1)
    metal_slope = pm.Normal('metallicity_slope', mu = 0, sd = 1)

    magnitudes = []

    metal_zp = pm.Normal('galaxy_zp', mu = -1.68, sd = 0.03)
    metal_coeff = pm.Normal('galaxy_slope', mu = 0.29, sd = 0.02)

    for i in range(len(log_periods)):

        if galaxy_metal[i] > -2.0:
            metal_mean = -2.0
        else:
            metal_mean = galaxy_metal[i]

        metal = pm.Normal(f'metallicity_{i}', mu = metal_mean, sd = 0.5, shape = star_nums[i])

        magnitudes.append(modulus[i] + zero_point + period_slope * log_periods[i] +
                          metal_slope * metal[star_ids[i]])

    calibrations = []

    for i in range(len(calibrate['Star Code'])):

        calibrations.append(field_moduli[i] + zero_point + period_slope * field_periods[i] +
                            metal_slope * field_metal[i])

    magnitudes.append(calibrations)
    modeled, observed = pm.math.concatenate(magnitudes), pm.math.concatenate(obs_mags)

    total_err = np.sqrt(sigma**2 + errors**2)

    obs = pm.Normal('obs', mu = modeled, sd = total_err, observed = observed)

map_estimate = pm.find_MAP(model = rr_lyrae_model)

with rr_lyrae_model:

    rr_lyrae_trace = pm.sample(cores = number_of_cpus, start = map_estimate)

pickle.dump(rr_lyrae_trace, open('rr_lyrae.pkl', 'wb'))
