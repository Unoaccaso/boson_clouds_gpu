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

from utils import generation, simulation

import matplotlib.pyplot as plt

import configparser

config = configparser.ConfigParser()
config.read(PATH_TO_THIS + "/config.ini")

import cupy
import numpy
from cupyx.profiler import benchmark

from utils.common import FLOAT_PRECISION, INT_PRECISION


def main():
    positions = generation.positions.get_positions()
    distances = generation.calculate_distances(positions)

    bh_masses, bh_spins, bh_ages_yrs = generation.sources.get_sources()
    boson_masses = generation.bosons.get_bosons()

    frequencies, amplitudes = simulation.get_signals(
        bh_masses,
        bh_ages_yrs,
        bh_spins,
        distances,
        boson_masses,
    )

    counts, bins = simulation.cupy_histograms(
        frequencies,
        amplitudes,
        100,
        10,
    )

    # idx = 700
    # plt.plot(cupy.asnumpy(bins)[idx], cupy.asnumpy(a=counts)[idx])


if __name__ == "__main__":
    # main()
    print(benchmark(main, n_repeat=10, n_warmup=5, name="gpu_v3.0"))
