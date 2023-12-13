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


def get_bosons():
    boson_distribution = distributions[
        config["simulation.parameters"]["BOSON_DISTRIBUTION"]
    ]
    boson_masses = boson_distribution(
        min=FLOAT_PRECISION(config["simulation.parameters"]["BOSON_MASS_MIN"]),
        max=FLOAT_PRECISION(config["simulation.parameters"]["BOSON_MASS_MAX"]),
        n_samples=INT_PRECISION(config["simulation.parameters"]["N_BOSONS"]),
    )
    return boson_masses
