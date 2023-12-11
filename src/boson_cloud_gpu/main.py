# ADDLICENSE


from utils import generate, analysis

import importlib


import matplotlib.pyplot as plt

import settings

importlib.reload(settings)

import cupy
import numpy


def main():
    # -> Compute the total number of sources from settings
    N_BHs_in_halo = (
        settings.SIMULATION["N_BHS_IN_HALO_CLUSTERS"]
        * settings.SIMULATION["N_CLUSTERS_IN_HALO"]
    )
    N_sources = settings.SIMULATION["N_BHS_IN_GALACTIC_CENTER"] + N_BHs_in_halo
    # ========= GENERATE SOURCES ===========
    #
    # ====== Masses =======
    BH_masses = generate.physics.Masses(
        N_sources,
        settings.SIMULATION["BH_MASS_MIN"],
        settings.SIMULATION["BH_MASS_MAX"],
    ).kroupa()

    # ====== Spins ======
    spins = generate.physics.Spins(
        N_sources,
        settings.SIMULATION["SPIN_MIN"],
        settings.SIMULATION["SPIN_MAX"],
    ).truncated_norm()

    # ===== Bosons =====
    boson_masses = generate.physics.Masses(
        settings.SIMULATION["N_BOSONS"],
        settings.SIMULATION["BOSON_MASS_MIN"],
        settings.SIMULATION["BOSON_MASS_MAX"],
    ).geomspace()

    # ======== Ages ========
    cluster_BH_ages = generate.physics.Ages(
        N_BHs_in_halo,
        settings.SIMULATION["HALO_BH_AGE_MIN"],
        settings.SIMULATION["HALO_BH_AGE_MAX"],
    ).geomuniform()
    # ).constant()
    core_BH_ages = generate.physics.Ages(
        settings.SIMULATION["N_BHS_IN_GALACTIC_CENTER"],
        settings.SIMULATION["CORE_BH_AGE_MIN"],
        settings.SIMULATION["CORE_BH_AGE_MAX"],
    ).geomuniform()
    # ).constant()
    # -> The order of concatenation is IMPORTANT.
    # BHs in core and halo have different characteristics
    BH_ages = cupy.concatenate((cluster_BH_ages, core_BH_ages))

    # -------------------------------------------------------------------------

    # ======= Positions ======
    # ========================
    # -> Sources from halo cluster relative positions inside the clusters
    halo_clusters = generate.Positions(
        n_points_per_cluster=int(settings.SIMULATION["N_BHS_IN_HALO_CLUSTERS"]),
        n_clusters=int(settings.SIMULATION["N_CLUSTERS_IN_HALO"]),
        cluster_3D_sizes=cupy.array([1, 1, 1]) * 10,  # Parsec
    ).gaussian()
    # -> Custer positions in the sky
    clusters_positions = generate.Positions(
        n_points_per_cluster=1,
        n_clusters=int(settings.SIMULATION["N_CLUSTERS_IN_HALO"]),
        cluster_3D_sizes=cupy.array([1, 1, 1])
        * float(settings.SIMULATION["GLOBULAR_CLUSTERS_DISTACE"]),
        empty_shape=True,
    ).uniform()
    # -> Sources from halo clusters absolute positions in the sky
    halo_sources_positions = halo_clusters + clusters_positions
    del halo_clusters, clusters_positions  # SAVING MEMORY

    # -> Generating sources from inside the galactic core
    core_sources_positions = generate.Positions(
        n_points_per_cluster=int(settings.SIMULATION["N_BHS_IN_GALACTIC_CENTER"])
    ).gaussian()

    # -> Merge all the positions in sky to a single array
    # Note that the first N positions will refer to halo BHs, the other will be
    # core BHs. This is important to construct the age array
    positions = generate.merge_position_clusters(
        halo_sources_positions, core_sources_positions
    )
    del halo_sources_positions, core_sources_positions

    # -> Compute distances
    distances = analysis.signal.distance(positions)
    del positions

    # ===========================================
    # ========== Compute Frequencies ============
    # ===========================================
    signal = analysis.Signal(
        boson_masses,
        BH_masses,
        BH_ages,
        spins,
        distances,
    )
    frequencies = signal.masked_frequencies
    amplitudes = signal.masked_amplitudes
    del signal

    """ n = 700
    frequencies = cupy.asnumpy(frequencies)
    plt.hist(frequencies[n][frequencies[n] > 0], 100)
    plt.show() """


if __name__ == "__main__":
    main()
