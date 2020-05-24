import argparse
import pandas as pd
import multiprocessing as mp

parser = argparse.ArgumentParser(description = "Helper for parallel processing.")
parser.add_argument('processes', metavar = 'N', type = int, nargs = 1, help = "Number of processes to use.")
number_of_cpus = parser.parse_args()

hubble = pd.read_csv("hubble.csv")

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
    return find_best_period(dataset, galaxy = row['Galaxy'], star = row['Star'])

pool = mp.Pool(processes = number_of_cpus)
hubble_results = pool.starmap(compute_period, [(row, hubble) for (index, row) in hubble.iterrows()])
pool.close()
