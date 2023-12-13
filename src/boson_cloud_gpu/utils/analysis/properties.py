import sys

import os.path

PATH_TO_THIS = os.path.dirname(__file__)
PATH_TO_MASTER = PATH_TO_THIS + "/../../"
sys.path.append(PATH_TO_MASTER)

import configparser

PATH_TO_SETTINGS = PATH_TO_MASTER + "/config.ini"
config = configparser.ConfigParser()
config.read(PATH_TO_SETTINGS)

simulation_constants = config["simulation.constants"]

from enum import Enum
import numpy

import astropy.constants


class FloatPrecision(Enum):
    FLOAT16 = numpy.float16
    FLOAT32 = numpy.float32
    FLOAT64 = numpy.float64


class IntPrecision(Enum):
    INT16 = numpy.int16
    INT32 = numpy.int32
    INT64 = numpy.int64


FLOAT_PRECISION = FloatPrecision[config["numeric.precision"]["FloatPrecision"]].value
INT_PRECISION = IntPrecision[config["numeric.precision"]["IntPrecision"]].value


class Constants(Enum):
    G = astropy.constants.G.value
    h_bar = astropy.constants.hbar.value
    M_sun = astropy.constants.M_sun.value
    h = astropy.constants.h.value
    # User defined
    om_0 = numpy.array(simulation_constants["OM0"], dtype=FLOAT_PRECISION)
    R0 = numpy.array(simulation_constants["R0"], dtype=FLOAT_PRECISION)
    T_obs = numpy.array(simulation_constants["TOBS"], dtype=INT_PRECISION)
    duty = numpy.array(simulation_constants["DUTY"], dtype=FLOAT_PRECISION)
    onev = numpy.array(simulation_constants["ONEV"], dtype=FLOAT_PRECISION)
