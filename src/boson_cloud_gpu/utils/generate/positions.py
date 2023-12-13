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
PATH_TO_MASTER = PATH_TO_THIS + "/../../"
sys.path.append(PATH_TO_MASTER)

import cupy
import configparser

from utils.distributions import distributions

PATH_TO_SETTINGS = PATH_TO_MASTER + "/config.ini"
config = configparser.ConfigParser()
config.read(PATH_TO_SETTINGS)


from utils.properties import FLOAT_PRECISION, INT_PRECISION


def is_supported(item: str, supported_list: list[str]):
    return item in supported_list


def get_positions():
    # black holes in galactic halo
    clusters_shape = (
        INT_PRECISION(config["simulation.parameters"]["N_BHS_IN_HALO_CLUSTERS"]),
        INT_PRECISION(config["simulation.parameters"]["N_CLUSTERS_IN_HALO"]),
    )
    halo_bh_relative_positions_distribution = distributions[
        config["simulation.parameters"]["HALO_RELATIVE_POSITION_DISTRIBUTION"]
    ]
    halo_clusters_positions_distribution = distributions[
        config["simulation.parameters"]["HALO_CLUSTER_POSITION_DISTRIBUTION"]
    ]
    halo_bh_relative_positions = halo_bh_relative_positions_distribution(
        shape=clusters_shape,
        radius=FLOAT_PRECISION(config["simulation.parameters"]["CLUSTERS_RADII"]),
    )
    halo_clusters_positions = halo_clusters_positions_distribution(
        shape=(1, INT_PRECISION(config["simulation.parameters"]["N_CLUSTERS_IN_HALO"])),
        radius=FLOAT_PRECISION(config["simulation.parameters"]["CLUSTER_DISTANCE"]),
    )
    halo_bh_absolute_positions = halo_bh_relative_positions + halo_clusters_positions

    # black holes in galactic core
    core_bh_positions_distribution = distributions[
        config["simulation.parameters"]["CORE_POSITION_DISTRIBUTION"]
    ]
    core_bh_positions = core_bh_positions_distribution(
        shape=(1, INT_PRECISION(config["simulation.parameters"]["N_BHS_IN_CORE"])),
        radius=FLOAT_PRECISION(config["simulation.parameters"]["CORE_RADIUS"]),
    )

    bh_positions = merge_position_clusters(
        halo_bh_absolute_positions, core_bh_positions
    )

    return bh_positions


def merge_position_clusters(cluster_1, cluster_2):
    cluster_1_pos_array = cupy.reshape(
        cluster_1,
        (
            cluster_1.shape[0] * cluster_1.shape[1],
            cluster_1.shape[2],
        ),
    )
    cluster_2_pos_array = cupy.reshape(
        cluster_2,
        (
            cluster_2.shape[0] * cluster_2.shape[1],
            cluster_2.shape[2],
        ),
    )

    return cupy.concatenate(
        [cluster_1_pos_array, cluster_2_pos_array],
        axis=0,
        dtype=FLOAT_PRECISION,
    )
