# ADDLICENSE


# %%
from lib import generate, analysis

import configparser


import cupy
import numpy


def main():
    # Importing settings from file
    settings = configparser.ConfigParser()
    settings.read("settings.ini")

    # Generate sky positions and compute distances
    distances = analysis.compute.random_distances(settings)

    # Generating sources
    N_sources = int(settings["SIMULATION"]["N_BHS_IN_GALACTIC_CENTER"]) + (
        int(settings["SIMULATION"]["N_BHS_IN_HALO_CLUSTERS"])
        * int(settings["SIMULATION"]["N_CLUSTERS_IN_HALO"])
    )
    BH_masses = generate.physics.Masses(
        N_sources,
        float(settings["SIMULATION"]["BH_MASS_MIN"]),
        float(settings["SIMULATION"]["BH_MASS_MAX"]),
    ).kroupa()
    boson_masses = generate.physics.Masses(
        N_sources,
        float(settings["SIMULATION"]["BOSON_MASS_MIN"]),
        float(settings["SIMULATION"]["BOSON_MASS_MAX"]),
    ).logspace()
    spins = generate.physics.Spins(
        N_sources,
        float(settings["SIMULATION"]["SPIN_MIN"]),
        float(settings["SIMULATION"]["SPIN_MAX"]),
    ).truncated_norm()


if __name__ == "__main__":
    main()
