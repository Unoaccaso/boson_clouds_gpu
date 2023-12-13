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

# from ._signal import Signal

import matplotlib.pyplot as plt

import configparser

config = configparser.ConfigParser()
config.read(PATH_TO_THIS + "/config.ini")

import cupy
import numpy
from cupyx.profiler import benchmark
import positions, sources, bosons

import my_signalV1

from properties import FLOAT_PRECISION, INT_PRECISION


def main():
    pos = positions.get_positions()
    distances = positions.calculate_distances(pos)

    bh_masses, bh_spins, bh_ages_yrs = sources.get_sources()
    boson_masses = bosons.get_bosons()

    sig = my_signalV1.SignalV1(
        boson_masses, bh_masses, bh_ages_yrs, bh_spins, distances
    )

    frequency = sig.masked_frequencies
    amplitude = sig.masked_amplitudes


if __name__ == "__main__":
    main()
    # print(benchmark(main, n_repeat=100, n_warmup=5, name="gpu_v1.0"))
