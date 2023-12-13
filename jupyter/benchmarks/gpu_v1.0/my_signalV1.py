# Copyright (C) 2023
# Riccardo Felicetti (felicettiriccardo1@gmail.com)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU AFFERO GENERAL PUBLIC LICENSE Version 3, 19 November 2007
#
# Copyright (C) 2007 Free Software Foundation, Inc. <https://fsf.org/>
# Everyone is permitted to copy and distribute verbatim copies
# of this license document, but changing it is not allowed.
#
# You should have received a copy of theGNU AFFERO GENERAL PUBLIC LICENSE
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


import sys
import os.path

PATH_TO_THIS = os.path.dirname(__file__)
PATH_TO_MASTER = PATH_TO_THIS
sys.path.append(PATH_TO_MASTER)

# standard packages
import cupy
from functools import cached_property

# user libraries
import configparser
import properties

PATH_TO_SETTINGS = PATH_TO_MASTER + "/config.ini"
config = configparser.ConfigParser()
config.read(PATH_TO_SETTINGS)


# global variables

BLOCK_SHAPE = (
    int(config["cuda"]["BlockSizeX"]),
    int(config["cuda"]["BlockSizeY"]),
)

FLOAT_PRECISION = properties.FloatPrecision[
    config["numeric.precision"]["FloatPrecision"]
].value
INT_PRECISION = properties.IntPrecision[
    config["numeric.precision"]["IntPrecision"]
].value


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
OM0 = FLOAT_PRECISION(config["simulation.parameters"]["OM0"])
R0 = FLOAT_PRECISION(config["simulation.parameters"]["R0"])
TOBS = FLOAT_PRECISION(config["simulation.parameters"]["TOBS"])
DUTY = FLOAT_PRECISION(config["simulation.parameters"]["DUTY"])
ONEV = FLOAT_PRECISION(config["simulation.parameters"]["ONEV"])


class SignalV1:
    def __init__(
        self,
        boson_mass,
        BH_mass,
        BH_ages_yrs,
        spins,
        distance,
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
        return self.frequency_at_detector

    @property
    def masked_frequencies(self):
        out_values = self.frequency_at_detector
        out_values[self.undetectable_values_mask] = cupy.nan
        return out_values

    @property
    def unmasked_amplitudes(self):
        return self.amplitude_at_detector

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
        ).astype(FLOAT_PRECISION)

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
        ).astype(FLOAT_PRECISION)
        emitted_frequency = (
            483
            * (self.boson_mass[:, cupy.newaxis] / 1e-12)
            * (1 - second_order_correction)
        ).astype(FLOAT_PRECISION)
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
