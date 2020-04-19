###################################################################################################
#
# Saha_Hybrid_Method.py
#
# Implementation of hybrid algorithm from Saha et al. (2017).
#
###################################################################################################

from pwkit.pdm import pdm
from astropy.timeseries import LombScargle

def phase_dispersion_minimization(times, magnitudes, uncertainties, frequencies, nbins = 10):
    lafler_kinman = pdm(times, magnitudes, uncertainties, 1 / frequencies, nbins)
    return lafler_kinman[0:3]

def lomb_scargle_analysis(times, magnitudes, uncertainties):
    frequency, power = LombScargle(times, magnitudes, uncertainties).autopower()
    return [frequency, power]

def hybrid_statistic(times, magnitudes, uncertainties):
    frequencies, pi = lomb_scargle_analysis(times, magnitudes, uncertainties)
    theta = phase_dispersion_minimization(times, magnitudes, uncertainties, frequencies)[0]
    return [frequencies, 2 * pi / theta]
