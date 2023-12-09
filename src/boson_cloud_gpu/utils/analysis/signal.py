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


class Signal:
    def __init__(
        self,
        boson_mass: NDArray,
        BH_mass: NDArray,
        BH_ages_yrs: NDArray,
        spins: NDArray,
        distance: NDArray,
        filter: bool = True,
    ) -> None:
        self._spins = spins
        self._filter = filter
        self._BH_mass = BH_mass
        self._boson_mass = boson_mass
        self._distance = distance

        # Compute all properties that have multiple calls in the code
        self.BH_ages_sec = BH_ages_yrs * 365 * 86_400

    @property
    def distance(self):
        return self._distance

    @property
    def boson_mass(self):
        return self._boson_mass

    @property
    def BH_mass(self):
        return self._BH_mass

    @property
    def spins(self):
        return self._spins

    @property
    def df_dot(self):
        #! TODO: CONTROLLARE CHE QUESTO DFDOT VA CALCOLATO SULLE FREQUENZE AL DETECTOR
        frequency_at_detector = self.frequency_at_detector
        return (
            OM0
            * cupy.sqrt(2 * cupy.ceil(frequency_at_detector / 10) * 10 * R0 / C)
            / (2 * TOBS / DUTY)
        )

    @property
    def unmasked_frequencies(self):
        return self.frequency_at_detector()

    @property
    def masked_frequencies(self):
        out_values = self.frequency_at_detector
        out_values[self.undetectable_values_mask] = cupy.nan
        return out_values

    @property
    def unmasked_amplitudes(self):
        return self.amplitude_at_detector()

    @property
    def masked_amplitudes(self):
        out_values = self.amplitude_at_detector
        out_values[self.undetectable_values_mask] = cupy.nan
        return out_values

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
            / 1e-24
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
    def amplitude_at_detector(self):
        alpha_7 = cupy.power(self.alpha / 0.1, 7)
        bh_mass = self.BH_mass
        spin = self.spins
        chi_c = self.chi_c

        timefactor = 1 + (self.BH_ages_sec - self.tau_inst) / self.tau_gw

        return (
            1 / cupy.sqrt(3) * 3.0e-24 / 10 * bh_mass * alpha_7 * (spin - chi_c) / 0.5
        ) / (timefactor * self.distance)

    @cached_property
    def undetectable_values_mask(self):
        frequency = self.frequency_at_detector
        tau_gw = self.tau_gw
        tau_inst = self.tau_inst
        is_detectable = cupy.array(
            (tau_gw * 3 > self.BH_ages_sec)
            & (self.alpha < 0.1)
            & (frequency > 20)
            & (frequency < 2048)
            & (tau_inst < 10 * self.BH_ages_sec)
            & (10 * tau_inst < tau_gw)
            & (self.spins > self.chi_c)
            & (self.df_dot > self.f_dot)
        )
        return cupy.logical_not(is_detectable)

    def plot(self):
        bh_ax, boson_ax = cupy.meshgrid(self.BH_mass, self.boson_mass)
