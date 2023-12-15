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
import cupy
import os.path


PATH_TO_THIS = os.path.dirname(__file__)
PATH_TO_MASTER = PATH_TO_THIS + "/../"
PATH_TO_KERNELS = PATH_TO_MASTER + "/utils/cuda_kernels/"
sys.path.append(PATH_TO_MASTER)

import configparser

PATH_TO_SETTINGS = PATH_TO_MASTER + "/config.ini"
config = configparser.ConfigParser()
config.read(PATH_TO_SETTINGS)

simulation_constants = config["simulation.constants"]

from enum import Enum
import cupy

import astropy.constants


class FloatPrecision(Enum):
    FLOAT16 = cupy.float16
    FLOAT32 = cupy.float32
    FLOAT64 = cupy.float64


class IntPrecision(Enum):
    INT16 = cupy.int16
    INT32 = cupy.int32
    INT64 = cupy.int64


FLOAT_PRECISION = FloatPrecision[config["numeric.precision"]["FloatPrecision"]].value
INT_PRECISION = IntPrecision[config["numeric.precision"]["IntPrecision"]].value


class Constants(Enum):
    G = astropy.constants.G.value
    h_bar = astropy.constants.hbar.value
    M_sun = astropy.constants.M_sun.value
    h = astropy.constants.h.value
    # User defined
    om_0 = cupy.array(simulation_constants["OM0"], dtype=FLOAT_PRECISION)
    R0 = cupy.array(simulation_constants["R0"], dtype=FLOAT_PRECISION)
    T_obs = cupy.array(simulation_constants["TOBS"], dtype=INT_PRECISION)
    duty = cupy.array(simulation_constants["DUTY"], dtype=FLOAT_PRECISION)
    onev = cupy.array(simulation_constants["ONEV"], dtype=FLOAT_PRECISION)


def get_preprocessing_module(module: str):
    # DOCUMENT THIS
    with open(PATH_TO_KERNELS + "/" + module, "r") as cuda_module_file:
        cuda_module = cupy.RawModule(code=cuda_module_file.read())
    return cuda_module
