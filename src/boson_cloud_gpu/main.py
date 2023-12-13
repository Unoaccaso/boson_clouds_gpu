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
    # # -> Compute the total number of sources from settings
    # N_BHs_in_halo = INT_PRECISION(
    #     config["simulation.parameters"]["N_BHS_IN_HALO_CLUSTERS"]
    # ) * INT_PRECISION(config["simulation.parameters"]["N_CLUSTERS_IN_HALO"])
    # N_sources = INT_PRECISION(
    #     INT_PRECISION(config["simulation.parameters"]["N_BHS_IN_CORE"]) + N_BHs_in_halo
    # )
    # # ========= GENERATE SOURCES ===========
    # #
    # # ====== Masses =======
    # BH_masses = generate.sources.CustomDistributions(
    #     N_sources,
    #     FLOAT_PRECISION(config["simulation.parameters"]["BH_MASS_MIN"]),
    #     FLOAT_PRECISION(config["simulation.parameters"]["BH_MASS_MAX"]),
    # ).kroupa()

    # # ====== Spins ======
    # BH_spins = generate.sources.Spins(
    #     N_sources,
    #     FLOAT_PRECISION(config["simulation.parameters"]["SPIN_MIN"]),
    #     FLOAT_PRECISION(config["simulation.parameters"]["SPIN_MAX"]),
    # ).truncated_norm()

    # # ===== Bosons =====
    # boson_masses = generate.sources.Masses(
    #     INT_PRECISION(config["simulation.parameters"]["N_BOSONS"]),
    #     FLOAT_PRECISION(config["simulation.parameters"]["BOSON_MASS_MIN"]),
    #     FLOAT_PRECISION(config["simulation.parameters"]["BOSON_MASS_MAX"]),
    # ).geomspace()

    # # ======== Ages ========
    # cluster_BH_ages = generate.sources.Ages(
    #     N_BHs_in_halo,
    #     FLOAT_PRECISION(config["simulation.parameters"]["HALO_BH_AGE_MIN"]),
    #     FLOAT_PRECISION(config["simulation.parameters"]["HALO_BH_AGE_MAX"]),
    # ).geomuniform()
    # # ).constant()
    # core_BH_ages = generate.sources.Ages(
    #     INT_PRECISION(config["simulation.parameters"]["N_BHS_IN_GALACTIC_CENTER"]),
    #     FLOAT_PRECISION(config["simulation.parameters"]["CORE_BH_AGE_MIN"]),
    #     FLOAT_PRECISION(config["simulation.parameters"]["CORE_BH_AGE_MAX"]),
    # ).geomuniform()
    # # ).constant()
    # # -> The order of concatenation is IMPORTANT.
    # # BHs in core and halo have different characteristics
    # BH_ages_yrs = cupy.concatenate(
    #     (cluster_BH_ages, core_BH_ages), dtype=FLOAT_PRECISION
    # )

    # # -------------------------------------------------------------------------

    # # ======= Positions ======
    # # ========================
    # # -> Sources from halo cluster relative positions inside the clusters
    # halo_clusters = generate.Positions(
    #     n_points_per_cluster=int(
    #         FLOAT_PRECISION(config["simulation.parameters"]["N_BHS_IN_HALO_CLUSTERS"])
    #     ),
    #     n_clusters=int(
    #         FLOAT_PRECISION(config["simulation.parameters"]["N_CLUSTERS_IN_HALO"])
    #     ),
    #     cluster_3D_sizes=cupy.array([1, 1, 1]) * 10,  # Parsec
    # ).gaussian()
    # # -> Custer positions in the sky
    # clusters_positions = generate.Positions(
    #     n_points_per_cluster=1,
    #     n_clusters=int(
    #         FLOAT_PRECISION(config["simulation.parameters"]["N_CLUSTERS_IN_HALO"])
    #     ),
    #     cluster_3D_sizes=cupy.array([1, 1, 1])
    #     * FLOAT_PRECISION(
    #         FLOAT_PRECISION(
    #             config["simulation.parameters"]["GLOBULAR_CLUSTERS_DISTACE"]
    #         )
    #     ),
    #     empty_shape=True,
    # ).uniform()
    # # -> Sources from halo clusters absolute positions in the sky
    # halo_sources_positions = halo_clusters + clusters_positions
    # del halo_clusters, clusters_positions  # SAVING MEMORY

    # # -> Generating sources from inside the galactic core
    # core_sources_positions = generate.Positions(
    #     n_points_per_cluster=int(
    #         FLOAT_PRECISION(config["simulation.parameters"]["N_BHS_IN_GALACTIC_CENTER"])
    #     )
    # ).gaussian()

    # # -> Merge all the positions in sky to a single array
    # # Note that the first N positions will refer to halo BHs, the other will be
    # # core BHs. This is important to construct the age array
    # positions = generate.merge_position_clusters(
    #     halo_sources_positions, core_sources_positions
    # )
    # del halo_sources_positions, core_sources_positions

    # # -> Compute distances
    # del positions

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
