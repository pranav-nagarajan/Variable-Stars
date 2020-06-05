import itertools
import argparse
import pickle
import multiprocessing as mp
import numpy as np
import pandas as pd
from gatspy import periodic
from astropy.timeseries import LombScargle

sparse_parser = argparse.ArgumentParser(description = "Helper for parallel processing.")
sparse_parser.add_argument('number_of_cpus', metavar = 'N', type = int, help = "Number of processes to use.")
sparse_parser.add_argument('photometric_data', type = str, help = "Light curve data for RR Lyrae stars.")
sparse_args = sparse_parser.parse_args()

data = pd.read_csv(sparse_args.photometric_data)
number_of_cpus = sparse_args.number_of_cpus

sparsities = np.array([1.0, 0.5, 0.25, 0.125])
sn_ratios = np.array([1, 10, 100, 1000])
all_combos = list(itertools.repeat(list(itertools.product(sparsities, sn_ratios)), 100))


def phase_dispersion_minimization(times, magnitudes, uncertainties, periods, weighted = True):
    """Implements the formula for calculating the Lafler-Kinman statistic
    using weighted phase dispersion minimization."""

    lafler_kinmans = []
    for period in periods:

        folded = (times / period) % 1
        ordered = sorted(list(zip(folded, magnitudes, uncertainties)), key = lambda x: x[0])
        unzipped = [list(t) for t in zip(*ordered)]
        measurements, errors = unzipped[1], unzipped[2]
        wrap_measurements = [measurements[-1]] + measurements
        wrap_errors = [errors[-1]] + errors

        weights = []
        for i in range(1, len(wrap_errors)):
            weights.append(1 / (wrap_errors[i]**2 + wrap_errors[i - 1]**2))

        numerator = []
        for j in range(1, len(wrap_measurements)):
            difference = (wrap_measurements[j] - wrap_measurements[j - 1])**2
            if weighted:
                numerator.append(difference * weights[j - 1])
            else:
                numerator.append(difference)

        if weighted:
            weighted_mean = np.average(np.array(measurements), weights = np.array(weights))
            denominator = sum(weights)*sum((np.array(measurements) - weighted_mean)**2)
        else:
            denominator = sum((np.array(measurements) - np.mean(measurements))**2)

        lafler_kinman = sum(numerator) / denominator
        lafler_kinmans.append(lafler_kinman)

    return np.array(lafler_kinmans)


def lomb_scargle_analysis(times, magnitudes, uncertainties, min_period = 0.2, max_period = 1.5, version = "astropy"):
    """Generates the Lomb-Scargle periodogram for a variable star light curve."""

    fit_periods = np.linspace(min_period, max_period, 100000)

    if version == "gatspy":
        model = periodic.LombScargleFast(fit_period = True)
        model.optimizer.period_range = (min_period, max_period)
        model.fit(times, magnitudes, uncertainties)
        results = model.score(fit_periods)
    else:
        astropy_model = LombScargle(times, magnitudes, uncertainties, normalization='psd', fit_mean=False)
        results = astropy_model.power(1 / fit_periods, method='slow')

    return [fit_periods, results]



def hybrid_statistic(times, magnitudes, uncertainties):
    """Computes the hybrid statistic defined by Saha et al. (2017).
    Then, uses the hybrid statistic to find the best period."""
    periods, pi = lomb_scargle_analysis(times, magnitudes, uncertainties)
    theta = phase_dispersion_minimization(times, magnitudes, uncertainties, periods)
    hybrid_statistic = np.array(2 * pi / theta)
    best_period = periods[np.argmax(hybrid_statistic)]
    return [1 / periods, pi, 2 / theta, hybrid_statistic, best_period]


def filter_data(dataset, passband, **kwargs):
    """Returns light curve data for a specific star in a specific passband."""

    filtered = dataset[dataset["Passband"] == passband]

    for elem in kwargs.keys():
        try:
            filtered = filtered[filtered[elem.capitalize()] == kwargs.get(elem)]
        except KeyError as e:
            print(f"Dataset does not contain {e.args[0]} column. Attempting to filter based on additional keyword arguments.")

    epoch = filtered["HJD-2400000.0"].values
    magnitudes = filtered["Magnitude"].values
    magnitude_errors = filtered["Uncertainty in Magnitude"].values
    return epoch, magnitudes, magnitude_errors


def plot_periodogram(dataset, passband, **kwargs):
    """Generates Lomb-Scargle, Lafler-Kinman, and hybrid periodograms for a variable star light curve."""

    epoch, magnitudes, magnitude_errors = filter_data(dataset, passband, **kwargs)
    frequencies, ls_powers, lk_powers, hybrid_powers, best_period = hybrid_statistic(epoch, magnitudes, magnitude_errors)

    return [frequencies, ls_powers, lk_powers, hybrid_powers, best_period]


def find_best_period(dataset, **kwargs):
    """Find the best period by averaging across the results from each passband."""

    passbands = dataset["Passband"].unique()
    freqs, ls_powers, lk_powers, hybrid_powers, period = plot_periodogram(dataset, passbands[0], **kwargs)
    print('\n')

    for passband in passbands[1:]:
        new_results = plot_periodogram(dataset, passband, **kwargs)
        ls_powers = ls_powers + new_results[1]
        lk_powers = lk_powers + new_results[2]
        hybrid_powers = hybrid_powers + new_results[3]
        print('\n')

    best_ls_period = 1 / (freqs[np.argmax(ls_powers)])
    best_lk_period = 1 / (freqs[np.argmax(lk_powers)])
    best_hybrid_period = 1 / (freqs[np.argmax(hybrid_powers)])

    return [best_ls_period, best_lk_period, best_hybrid_period]


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
        sparse_periods.append(simulate_best_period(data, combo[0], combo[1]))
    return sparse_periods


pool = mp.Pool(processes = number_of_cpus)
sparse_results = pool.map(sparse_periods, all_combos)
pool.close()

pickle.dump(np.array(sparse_results), open("sparse_noisy_periods.pkl", "wb"))
