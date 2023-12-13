# ADDLICENSE

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


if __name__ == "__main__":
    main()
