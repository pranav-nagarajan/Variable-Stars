import itertools
import argparse
import pickle
import multiprocessing as mp
import numpy as np

parser = argparse.ArgumentParser(description = "Helper for parallel processing.")
parser.add_argument('number_of_cpus', metavar = 'N', type = int, help = "Number of processes to use.")
args = parser.parse_args()
number_of_cpus = args.number_of_cpus

sparsities = np.array([1.0, 0.5, 0.25, 0.125])
sn_ratios = np.array([1, 10, 100, 1000])
all_combos = list(itertools.repeat(list(itertools.product(sparsities, sn_ratios)), 100))

def sparse_periods(combinations):
    sparse_periods = []
    for combo in combinations:
        sparse_periods.append(simulate_best_period(saha, combo[0], combo[1]))
    return sparse_periods

pool = mp.Pool(processes = number_of_cpus)
sparse_results = pool.map(sparse_periods, all_combos)
pool.close()

pickle.dump(np.array(sparse_results), open("sparse_noisy_periods.pkl", "wb"))
