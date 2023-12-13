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

from utils import generate, simulate

import matplotlib.pyplot as plt

import configparser

config = configparser.ConfigParser()
config.read(PATH_TO_THIS + "/config.ini")

import cupy
import numpy
from cupyx.profiler import benchmark

from utils.properties import FLOAT_PRECISION, INT_PRECISION


def main():
    positions = generate.positions.get_positions()
    distances = simulate.calculate_distances(positions)

    bh_masses, bh_spins, bh_ages_yrs = generate.sources.get_sources()
    boson_masses = generate.bosons.get_bosons()

    frequencies, amplitudes = simulate.get_signals(
        bh_masses,
        bh_ages_yrs,
        bh_spins,
        distances,
        boson_masses,
    )
    arr = cupy.asnumpy(amplitudes)[7000]
    bins = numpy.geomspace(numpy.nanmin(arr), numpy.nanmax(arr), 100)
    plt.hist(arr, bins)
    plt.xscale("log")
    plt.show()


if __name__ == "__main__":
    main()
    # print(benchmark(main, n_repeat=10))
