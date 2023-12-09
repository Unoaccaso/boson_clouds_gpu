# ADDLICENSE

from .. import generate

import sys
from functools import cached_property

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
ONEV = settings.CONSTANTS["ONEV"]
PRECISION = settings.GENERAL["PRECISION"]


def distance(positions: NDArray):
    # Computing distances
    distances = cupy.sqrt(
        positions[:, 0] * positions[:, 0]
        + positions[:, 1] * positions[:, 1]
        + positions[:, 2] * positions[:, 2]
    )
    del positions

    return distances


class Frequency:
    def __init__(
        self,
        boson_mass: NDArray,
        BH_mass: NDArray,
        BH_ages_yrs: NDArray,
        spins: NDArray,
        filter: bool = True,
    ) -> None:
        self._spins = spins
        self._filter = filter
        self._BH_mass = BH_mass
        self._boson_mass = boson_mass

        # Compute all properties that have multiple calls in the code
        self.BH_ages_sec = BH_ages_yrs * 365 * 86_400

    @property
    def boson_mass(self):
        return self._boson_mass

    @property
    def BH_mass(self):
        return self._BH_mass

    @property
    def spins(self):
        return self._spins

    @cached_property
    def BH_mass_2(self):
        return cupy.power(self._BH_mass, 2)

    @cached_property
    def boson_mass_2(self):
        return cupy.power(self._boson_mass, 2)

    @cached_property
    def alpha(self):
        return (
            G / (C**3 * HBAR) * 2e30 * self.BH_mass * self.boson_mass[:, None] * ONEV
        ).astype(PRECISION)

    @cached_property
    def f_dot(self):
        alpha = self.alpha
        _alpha_17 = cupy.power(alpha / 0.1, 17)
        return (
            (7e-15 + 1e-10 * (1e17 / 1e30) ** 4)
            * self.boson_mass_2[:, cupy.newaxis]
            * _alpha_17
        )

    @cached_property
    def tau_inst(self):
        alpha = self.alpha
        _alpha_9 = cupy.power(alpha / 0.1, 9)
        return 27 * 86400 / 10.0 * self.BH_mass * (1 / _alpha_9) / self.spins

    @cached_property
    def tau_gw(self):
        _alpha_15 = cupy.power(self.alpha / 0.1, 15)
        return 6.5e4 * 365 * 86400 * self.BH_mass / 10 * (1 / _alpha_15) / self.spins

    @cached_property
    def chi_c(self):
        return 4 * self.alpha / (1 + 4.0 * self.alpha**2)

    @cached_property
    def frequency_at_detector(self):
        second_order_correction = (
            0.0056 / 8 * 1e22 * (self.BH_mass_2 * self.boson_mass_2[:, cupy.newaxis])
        ).astype(PRECISION)
        emitted_frequency = (
            483
            * (self.boson_mass[:, cupy.newaxis] / 1e-12)
            * (1 - second_order_correction)
        ).astype(PRECISION)
        f_dot = self.f_dot
        tau_inst = self.tau_inst

        frequency_at_detector = emitted_frequency + f_dot * (
            self.BH_ages_sec - tau_inst
        )
        return frequency_at_detector

    @cached_property
    def df_dot(self):
        #! TODO: CONTROLLARE CHE QUESTO DFDOT VA CALCOLATO SULLE FREQUENZE AL DETECTOR
        frequency_at_detector = self.frequency_at_detector
        return (
            OM0
            * cupy.sqrt(2 * cupy.ceil(frequency_at_detector / 10) * 10 * R0 / C)
            / (2 * TOBS / DUTY)
        )

    @property
    def unmasked_values(self):
        return self.compute()

    @property
    def masked_values(self):
        values = self.frequency_at_detector
        tau_gw = self.tau_gw
        tau_inst = self.tau_inst
        bool_arr = cupy.array(
            (tau_gw > 3 * self.BH_ages_sec)
            & (self.alpha < 0.1)
            & (values > 20)
            & (values < 2048)
            & (tau_inst < 10 * self.BH_ages_sec)
            & (10 * tau_inst < tau_gw)
            & (self.spins > self.chi_c)
            & (self.df_dot > self.f_dot)
        )
        mask = cupy.logical_not(bool_arr)
        values[mask] = cupy.nan
        return values

    def plot(self):
        bh_ax, boson_ax = cupy.meshgrid(self.BH_mass, self.boson_mass)


def frequencies(
    boson_mass: NDArray,
    BH_mass: NDArray,
    BH_ages_yrs: NDArray,
    spins: NDArray,
    filter: bool = True,
):
    _alpha = alpha(BH_mass, boson_mass)

    boson_mass_2 = cupy.power(boson_mass, 2)
    BH_mass_2 = cupy.power(BH_mass, 2)
    frequency = (
        483
        * (boson_mass[:, cupy.newaxis] / 1e-12)
        * (1 - 0.0056 / 8 * 1e22 * (BH_mass_2 * boson_mass_2[:, cupy.newaxis]))
    ).astype(PRECISION)
    del BH_mass_2

    """ _alpha_17 = cupy.power(_alpha / 0.1, 17)
    _fdot = (
        (7e-15 + 1e-10 * (1e17 / 1e30) ** 4) * boson_mass_2[:, cupy.newaxis] * _alpha_17
    )
    del boson_mass_2, _alpha_17 """

    _f_dot = fdot(BH_mass, boson_mass)
    _alpha_9 = cupy.power(_alpha / 0.1, 9)
    tau_inst = 27 * 86400 / 10.0 * BH_mass * (1 / _alpha_9) / spins
    del _alpha_9

    BH_ages_sec = BH_ages_yrs * 86_400 * 365
    frequency_at_detector = frequency + _f_dot * (BH_ages_sec - tau_inst)
    del frequency

    """ maximum_frequency = (
        C**3
        / (2 * cupy.pi * G * 2e30 * BH_mass)
        * spins
        / (1 + cupy.sqrt(1 - spins**2))
    ) """

    _alpha_15 = cupy.power(_alpha / 0.1, 15)
    tau_gw = 6.5e4 * 365 * 86400 * BH_mass / 10 * (1 / _alpha_15) / spins
    del _alpha_15

    # -> chi_c
    chi_c = 4 * _alpha / (1 + 4.0 * _alpha**2)
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
            & (_alpha < 0.1)
            & (frequency_at_detector > 20)
            & (frequency_at_detector < 2048)
            & (tau_inst < 10 * BH_ages_sec)
            & (10 * tau_inst < tau_gw)
            & (spins > chi_c)
            & (dfdot > _f_dot)
        )
        mask = cupy.logical_not(bool_arr)
        frequency_at_detector[mask] = cupy.nan

    return frequency_at_detector.astype(PRECISION)


def amplitude(
    boson_mass: NDArray,
    BH_mass: NDArray,
    BH_ages_yrs: NDArray,
    spins: NDArray,
    settings: dict,
    filter: bool = True,
):
    ...


def alpha(BH_mass, boson_mass):
    _alpha = (
        G / (C**3 * HBAR) * 2e30 * BH_mass * boson_mass[:, cupy.newaxis] * ONEV
    ).astype(PRECISION)
    return _alpha


def fdot(BH_mass, boson_mass):
    _alpha = alpha(BH_mass, boson_mass)
    _alpha_17 = cupy.power(_alpha / 0.1, 17)
    boson_mass_2 = cupy.power(boson_mass, 2)
    BH_mass_2 = cupy.power(BH_mass, 2)
    return (
        (7e-15 + 1e-10 * (1e17 / 1e30) ** 4) * boson_mass_2[:, cupy.newaxis] * _alpha_17
    )
