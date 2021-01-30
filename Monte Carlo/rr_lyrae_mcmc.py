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
galaxy_mags = galaxies['Apparent V Magnitude'].values
galaxy_mag_err = galaxies['Error in Magnitude'].values
galaxy_wesenheit = galaxies['Wesenheit Magnitude'].values

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

field_mags_BV = calibrate['B-V Wesenheit Magnitude'].values
field_mags_VI = calibrate['V-I Wesenheit Magnitude'].values
obs_mags.append(field_mags_BV)
obs_mags.append(field_mags_VI)

field_mag_err_BV = calibrate['Uncertainty in B-V Wesenheit Magnitude'].values
field_mag_err_VI = calibrate['Uncertainty in V-I Wesenheit Magnitude'].values
field_mod_err = calibrate['Uncertainty in Distance Modulus'].values
errors.append(np.sqrt(field_mag_err_BV**2 + field_mod_err**2))
errors.append(np.sqrt(field_mag_err_VI**2 + field_mod_err**2))
errors = np.hstack(errors)

rr_lyrae_model = pm.Model()

with rr_lyrae_model:

    sigma_BV = pm.HalfNormal('sigma_BV', sd = 0.5)
    sigma_VI = pm.HalfNormal('sigma_VI', sd = 0.5)

    modulus = pm.Normal('modulus', mu = 20, sd = 10, shape = len(lin_reg_tables))

    zero_point_BV = pm.Normal('zero_point_BV', mu = 0, sd = 1)
    period_slope_BV = pm.Normal('period_slope_BV', mu = 0, sd = 1)
    metal_slope_BV = pm.Normal('metallicity_slope_BV', mu = 0, sd = 1)

    zero_point_VI = pm.Normal('zero_point_VI', mu = 0, sd = 1)
    period_slope_VI = pm.Normal('period_slope_VI', mu = 0, sd = 1)
    metal_slope_VI = pm.Normal('metallicity_slope_VI', mu = 0, sd = 1)

    magnitudes = []
    sigmas = []

    metal_zp = pm.Normal('galaxy_zp', mu = -1.68, sd = 0.03)
    metal_coeff = pm.Normal('galaxy_slope', mu = 0.29, sd = 0.02)

    for i in range(len(log_periods)):

        galaxy_mag = pm.Normal(f'magnitude_{i}', mu = galaxy_mags[i], sd = galaxy_mag_err[i])

        log_term = -6 - 0.4 * (galaxy_mag - modulus[i] - 4.83)
        metal_mean = metal_zp + metal_coeff * log_term
        metal = pm.Normal(f'metallicity_{i}', mu = metal_mean, sd = 0.5, shape = star_nums[i])

        if galaxy_wesenheit[i] == 'B-V':
            magnitudes.append(modulus[i] + zero_point_BV + period_slope_BV * log_periods[i] +
                              metal_slope_BV * metal[star_ids[i]])
            sigmas.append(sigma_BV)
        else:
            magnitudes.append(modulus[i] + zero_point_VI + period_slope_VI * log_periods[i] +
                              metal_slope_VI * metal[star_ids[i]])
            sigmas.append(sigma_VI)

    calibrations = []

    for i in range(len(calibrate['Star Code'])):

        calibrations.append(field_moduli[i] + zero_point_BV + period_slope_BV * field_periods[i] +
                            metal_slope_BV * field_metal[i])
        sigmas.append(sigma_BV)

    for i in range(len(calibrate['Star Code'])):

        calibrations.append(field_moduli[i] + zero_point_VI + period_slope_VI * field_periods[i] +
                            metal_slope_VI * field_metal[i])
        sigmas.append(sigma_VI)

    magnitudes.append(calibrations)
    modeled, observed = pm.math.concatenate(magnitudes), pm.math.concatenate(obs_mags)

    sigmas = tt.as_tensor_variable(sigmas)
    total_err = np.sqrt(sigmas**2 + errors**2)

    obs = pm.Normal('obs', mu = modeled, sd = total_err, observed = observed)

map_estimate = pm.find_MAP(model = rr_lyrae_model)

with rr_lyrae_model:

    rr_lyrae_trace = pm.sample(cores = number_of_cpus, start = map_estimate)

pickle.dump(rr_lyrae_trace, open('rr_lyrae.pkl', 'wb'))
