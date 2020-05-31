import argparse
import pickle
import numpy as np
import pandas as pd
import multiprocessing as mp
from gatspy import periodic
from astropy.timeseries import LombScargle

parser = argparse.ArgumentParser(description = "Helper for parallel processing.")
parser.add_argument('number_of_cpus', metavar = 'N', type = int, help = "Number of processes to use.")
parser.add_argument('photometric_data', type = str, help = "Light curve data for RR Lyrae stars.")
parser.add_argument('star_catalog', type = str, help = "Catalog of unique RR Lyrae stars.")
args = parser.parse_args()

number_of_cpus = args.number_of_cpus
photometric_data = pd.read_csv(args.photometric_data)
star_catalog = pd.read_csv(args.star_catalog)

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
            weighted_mean = np.mean(np.array(measurements) * np.array(weights))
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


def compute_period(row, dataset):
    if 'Galaxy' in row.keys() and 'Star' in row.keys():
        galaxy = row['Galaxy']
        star = row['Star']
        return find_best_period(dataset, galaxy = galaxy, star = star)
    elif 'Star' in row.keys():
        star = row['Star']
        return find_best_period(dataset, star = star)
    else:
        return find_best_period(dataset)


pool = mp.Pool(processes = number_of_cpus)
iterables = [(row, photometric_data) for (index, row) in star_catalog.iterrows()]
hubble_results = pool.starmap(compute_period, iterables)
pool.close()

pickle.dump(hubble_results, open("computed_periods.pkl", "wb"))
