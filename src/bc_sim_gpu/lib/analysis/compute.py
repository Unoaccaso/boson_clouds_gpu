# ADDLICENSE

from .. import generate

import configparser
import cupy


def random_distances(config):
    # Generating the sources of signals
    galactic_halo_clusters_centers = generate.Positions(
        n_points_per_cluster=int(config["SIMULATION"]["N_BHS_IN_HALO_CLUSTERS"]),
        n_clusters=int(config["SIMULATION"]["N_CLUSTERS_IN_HALO"]),
        cluster_3D_sizes=cupy.array([1, 1, 1]) * 10,
    ).gaussian()

    clusters_center_positions = generate.Positions(
        n_points_per_cluster=1,
        n_clusters=int(config["SIMULATION"]["N_CLUSTERS_IN_HALO"]),
        cluster_3D_sizes=cupy.array([1, 1, 1])
        * float(config["SIMULATION"]["GLOBULAR_CLUSTERS_DISTACE"]),
        empty_shape=True,
    ).uniform()

    halo_sources_positions = galactic_halo_clusters_centers + clusters_center_positions
    del galactic_halo_clusters_centers, clusters_center_positions

    halo_sources_positions_list = cupy.reshape(
        halo_sources_positions,
        (
            halo_sources_positions.shape[0] * halo_sources_positions.shape[1],
            halo_sources_positions.shape[2],
        ),
    )
    del halo_sources_positions

    gal_center_sources_positions = generate.Positions(
        n_points_per_cluster=int(config["SIMULATION"]["N_BHS_IN_GALACTIC_CENTER"])
    ).gaussian()
    gal_center_sources_positions_list = cupy.reshape(
        gal_center_sources_positions,
        (
            gal_center_sources_positions.shape[0]
            * gal_center_sources_positions.shape[1],
            gal_center_sources_positions.shape[2],
        ),
    )
    del gal_center_sources_positions

    sources_positions = cupy.concatenate(
        [halo_sources_positions_list, gal_center_sources_positions_list], axis=0
    )
    del gal_center_sources_positions_list, halo_sources_positions_list

    # Computing distances
    distances = cupy.sqrt(
        sources_positions[:, 0] * sources_positions[:, 0]
        + sources_positions[:, 1] * sources_positions[:, 1]
        + sources_positions[:, 2] * sources_positions[:, 2]
    )
    del sources_positions

    return distances
