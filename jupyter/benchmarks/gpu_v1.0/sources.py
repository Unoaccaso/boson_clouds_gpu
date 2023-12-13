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
PATH_TO_MASTER = PATH_TO_THIS
sys.path.append(PATH_TO_MASTER)

import cupy
import configparser

from distributions import distributions

PATH_TO_SETTINGS = PATH_TO_MASTER + "/config.ini"
config = configparser.ConfigParser()
config.read(PATH_TO_SETTINGS)

from properties import FLOAT_PRECISION, INT_PRECISION


def get_sources():
    # generate halo sources
    n_bh_in_halo = INT_PRECISION(
        INT_PRECISION(config["simulation.parameters"]["N_BHS_IN_HALO_CLUSTERS"])
        * INT_PRECISION(config["simulation.parameters"]["N_CLUSTERS_IN_HALO"])
    )
    halo_mass_distribution = distributions[
        config["simulation.parameters"]["HALO_MASS_DISTRIBUTION"]
    ]
    halo_spin_distribution = distributions[
        config["simulation.parameters"]["HALO_SPIN_DISTRIBUTION"]
    ]
    halo_age_distribution = distributions[
        config["simulation.parameters"]["HALO_AGE_DISTRIBUTION"]
    ]
    halo_bh_masses = halo_mass_distribution(
        min=FLOAT_PRECISION(config["simulation.parameters"]["HALO_BH_MASS_MIN"]),
        max=FLOAT_PRECISION(config["simulation.parameters"]["HALO_BH_MASS_MAX"]),
        n_samples=n_bh_in_halo,
    )
    halo_bh_spins = halo_spin_distribution(
        min=FLOAT_PRECISION(config["simulation.parameters"]["HALO_SPIN_MIN"]),
        max=FLOAT_PRECISION(config["simulation.parameters"]["HALO_SPIN_MAX"]),
        n_samples=n_bh_in_halo,
    )
    halo_bh_ages = halo_age_distribution(
        min=FLOAT_PRECISION(config["simulation.parameters"]["HALO_BH_AGE_MIN"]),
        max=FLOAT_PRECISION(config["simulation.parameters"]["HALO_BH_AGE_MAX"]),
        n_samples=n_bh_in_halo,
    )
    # generate galaxy core surces
    core_mass_distribution = distributions[
        config["simulation.parameters"]["CORE_MASS_DISTRIBUTION"]
    ]
    core_spin_distribution = distributions[
        config["simulation.parameters"]["CORE_SPIN_DISTRIBUTION"]
    ]
    core_age_distribution = distributions[
        config["simulation.parameters"]["CORE_AGE_DISTRIBUTION"]
    ]
    core_bh_masses = core_mass_distribution(
        min=FLOAT_PRECISION(config["simulation.parameters"]["CORE_BH_MASS_MIN"]),
        max=FLOAT_PRECISION(config["simulation.parameters"]["CORE_BH_MASS_MAX"]),
        n_samples=INT_PRECISION(config["simulation.parameters"]["N_BHS_IN_CORE"]),
    )
    core_bh_spins = core_spin_distribution(
        min=FLOAT_PRECISION(config["simulation.parameters"]["CORE_SPIN_MIN"]),
        max=FLOAT_PRECISION(config["simulation.parameters"]["CORE_SPIN_MAX"]),
        n_samples=INT_PRECISION(config["simulation.parameters"]["N_BHS_IN_CORE"]),
    )
    core_bh_ages = core_age_distribution(
        min=FLOAT_PRECISION(config["simulation.parameters"]["CORE_BH_AGE_MIN"]),
        max=FLOAT_PRECISION(config["simulation.parameters"]["CORE_BH_AGE_MAX"]),
        n_samples=INT_PRECISION(config["simulation.parameters"]["N_BHS_IN_CORE"]),
    )

    bh_masses = cupy.concatenate(
        (halo_bh_masses, core_bh_masses), dtype=FLOAT_PRECISION
    )
    bh_spins = cupy.concatenate((halo_bh_spins, core_bh_spins), dtype=FLOAT_PRECISION)

    bh_ages = cupy.concatenate((halo_bh_ages, core_bh_ages), dtype=FLOAT_PRECISION)

    return (bh_masses, bh_spins, bh_ages)
