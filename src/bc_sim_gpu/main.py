# ADDLICENSE


# %%
from lib import generate, analysis

import configparser


import cupy
import numpy


def main():
    # Importing settings from file
    config = configparser.ConfigParser()
    config.read("settings.ini")

    # Computing distances
    distances = analysis.compute.random_distances(config)


if __name__ == "__main__":
    main()
