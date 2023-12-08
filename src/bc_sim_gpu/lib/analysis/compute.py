# ADDLICENSE

from .. import generate

import sys

sys.path.append("../../")
import settings

import cupy
from cupy.typing import NDArray

from numba import cuda

from astropy.constants import (
    G,
    hbar,
    M_sun,
    c,
    h,
)

G = G.value
HBAR = hbar.value
C = c.value
M_SUN = M_sun.value
H = h.value
OM0 = settings.CONSTANTS["OM0"]
R0 = settings.CONSTANTS["R0"]
TOBS = settings.CONSTANTS["TOBS"]
DUTY = settings.CONSTANTS["DUTY"]


def distance(positions: NDArray):
    # Computing distances
    distances = cupy.sqrt(
        positions[:, 0] * positions[:, 0]
        + positions[:, 1] * positions[:, 1]
        + positions[:, 2] * positions[:, 2]
    )
    del positions

    return distances


def frequencies(
    boson_mass: NDArray,
    BH_mass: NDArray,
    BH_ages_yrs: NDArray,
    spins: NDArray,
    settings: dict,
    filter: bool = True,
):
    c_3 = C * C * C  # = c^3

    alpha = (
        G
        / (c_3 * HBAR)
        * 2e30
        * BH_mass
        * boson_mass[:, cupy.newaxis]
        * settings.CONSTANTS["ONEV"]
    ).astype(settings.GENERAL["PRECISION"])

    boson_mass_2 = cupy.power(boson_mass / 1e-12, 2)
    BH_mass_2 = cupy.power(BH_mass / 10, 2)
    frequency = (
        483
        * (boson_mass[:, cupy.newaxis] / 1e-12)
        * (1 - 0.0056 / 8 * BH_mass_2 * boson_mass_2[:, cupy.newaxis])
    )
    del BH_mass_2

    alpha_17 = cupy.power(alpha / 0.1, 17)
    f_dot = 7e-15 * boson_mass_2[:, cupy.newaxis] * alpha_17
    f_dot_2 = 1e-10 * (1e17 / 1e30) ** 4 * boson_mass_2[:, cupy.newaxis] * alpha_17
    del boson_mass_2, alpha_17

    f_dot = f_dot + f_dot_2
    del f_dot_2

    alpha_9 = cupy.power(alpha / 0.1, 9)
    # -> tau_inst
    tau_inst = 27 * 86400 / 10.0 * BH_mass * (1 / alpha_9) / spins
    del alpha_9

    BH_ages_sec = BH_ages_yrs * 86_400 * 365
    frequency_at_detector = frequency + f_dot * (BH_ages_sec - tau_inst)

    maximum_frequency = (
        c_3
        / (2 * cupy.pi * G * 2e30 * BH_mass)
        * spins
        / (1 + cupy.sqrt(1 - spins**2))
    )

    alpha_15 = cupy.power(alpha / 0.1, 15)
    # -> tau_gw
    tau_gw = 6.5e4 * 365 * 86400 * BH_mass / 10 * (1 / alpha_15) / spins
    del alpha_15

    # -> chi_c
    chi_c = 4 * alpha / (1 + 4.0 * alpha**2)
    dfdot = (
        OM0
        * cupy.sqrt(2 * cupy.ceil(frequency_at_detector / 10) * 10 * R0 / C)
        / (2 * TOBS / DUTY)
    )

    # Mask undetectable signals
    if filter:
        """
        conditions to be met in order to have a potentially detectable signal
        (there may be some redundance)

        o tau_inst < t0s          : superradiance time scale must be shorter than system age
        o freq < freq_max         : condition for the development of the instability
        o 10*tau_inst < tau_gw    : we want the instability is fully completed
        o chi_i > chi_c           : condition for the development of the instability
        o (freq>20) & (freq<610)  : GW frequency in the search band
        o dfdot > fdot            : signal spin-up within half bin
        """
        bool_arr = cupy.array(
            (tau_gw > 3 * BH_ages_sec)
            & (alpha < 0.1)
            & (frequency_at_detector > 20)
            & (frequency_at_detector < 2048)
            & (tau_inst < 10 * BH_ages_sec)
            & (10 * tau_inst < tau_gw)
            & (spins > chi_c)
            & (dfdot > f_dot)
        )
        mask = cupy.logical_not(bool_arr)
        frequency_at_detector[mask] = cupy.nan

    return frequency_at_detector


def amplitude():
    ...
