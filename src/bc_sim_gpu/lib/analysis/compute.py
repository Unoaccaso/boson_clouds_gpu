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
):
    c_3 = C * C * C  # = c^3

    alpha = (
        G
        / (c_3 * HBAR)
        * 2e30
        * BH_mass
        * boson_mass[:, cupy.newaxis]
        * float(settings.CONSTANTS["ONEV"])
    ).astype(cupy.float32)

    boson_mass_2 = cupy.power(boson_mass * 1e-12, 2)
    BH_mass_2 = cupy.power(BH_mass / 10, 2)
    frequency = (
        483
        * (boson_mass[:, cupy.newaxis] / 1e-12)
        * (1 - 7e-4 * BH_mass_2 * boson_mass_2[:, cupy.newaxis])
    )
    del BH_mass_2

    alpha_17 = cupy.power(alpha / 0.1, 17)
    f_dot = 7e-15 * boson_mass_2[:, cupy.newaxis] * alpha_17
    f_dot_2 = 1e-10 * (1e17 / 1e30) ** 4 * boson_mass_2[:, cupy.newaxis] * alpha_17
    del boson_mass_2, alpha_17

    f_dot = f_dot + f_dot_2
    del f_dot_2

    alpha_9 = cupy.power(alpha / 0.1, 9)
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
    tau_gw = 6.5e4 * 365 * 86400 * BH_mass / 10 * (1 / alpha_15) / spins
    del alpha_15

    # TODO implementare la selezione sulle frequenze visibili
    return frequency_at_detector


def amplitude():
    ...
