import itertools
import argparse
import pickle
import multiprocessing as mp
import numpy as np
from compute_period import (phase_dispersion_minimization, lomb_scargle_analysis, hybrid_statistic,
filter_data, plot_periodogram, find_best_period)

sparse_parser = argparse.ArgumentParser(description = "Helper for parallel processing.")
sparse_parser.add_argument('number_of_cpus', metavar = 'N', type = int, help = "Number of processes to use.")
sparse_args = sparse_parser.parse_args()
number_of_cpus = sparse_args.number_of_cpus

sparsities = np.array([1.0, 0.5, 0.25, 0.125])
sn_ratios = np.array([1, 10, 100, 1000])
all_combos = list(itertools.repeat(list(itertools.product(sparsities, sn_ratios)), 100))


def simulate_sparsity_and_noise(dataset, passband, sparsity = 1.0, signal_to_noise = 0, **kwargs):
    """Simulates sparsity and noise in the light curve data of a specific star in a specific passband."""

    times, magnitudes, errors = filter_data(dataset, passband, **kwargs)
    simulated = pd.DataFrame({"HJD-2400000.0" : times, "Magnitude" : magnitudes,
                              "Uncertainty in Magnitude" : errors, "Passband" : passband})

    if sparsity != 1.0:
        simulated = simulated.sample(frac = sparsity)

    if signal_to_noise != 0:
        simulated["Uncertainty in Magnitude"] = 2.5 * np.log10(1 + 1 / signal_to_noise)
        noise = [np.random.normal(0, error) for error in simulated["Uncertainty in Magnitude"].values]
        simulated["Magnitude"] = simulated["Magnitude"] + noise

    return simulated


def simulate_best_period(dataset, sparsity = 1.0, signal_to_noise = 0, **kwargs):
    """Simulates sparsity and noise in the light curve data of a specific star.
    Then, determines the effect on the calculation of the best period."""

    passbands = dataset["Passband"].unique()
    initial = simulate_sparsity_and_noise(dataset, passbands[0], sparsity, signal_to_noise, **kwargs)

    for passband in passbands[1:]:

        new_data = simulate_sparsity_and_noise(dataset, passband, sparsity, signal_to_noise, **kwargs)
        initial = pd.concat([initial, new_data], axis = 0)

    best_periods = find_best_period(initial, **kwargs)
    return best_periods


def sparse_periods(combinations):
    sparse_periods = []
    for combo in combinations:
        sparse_periods.append(simulate_best_period(saha, combo[0], combo[1]))
    return sparse_periods


pool = mp.Pool(processes = number_of_cpus)
sparse_results = pool.map(sparse_periods, all_combos)
pool.close()

pickle.dump(np.array(sparse_results), open("sparse_noisy_periods.pkl", "wb"))
