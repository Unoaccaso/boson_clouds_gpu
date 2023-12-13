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
sys.path.append(PATH_TO_THIS)

import cupy
import numpy
import warnings
import configparser

import properties

PATH_TO_SETTINGS = PATH_TO_THIS + "/config.ini"
config = configparser.ConfigParser()
config.read(PATH_TO_SETTINGS)

FLOAT_PRECISION = properties.FloatPrecision[
    config["numeric.precision"]["FloatPrecision"]
].value
INT_PRECISION = properties.IntPrecision[
    config["numeric.precision"]["IntPrecision"]
].value

from typing import NamedTuple


def uniform(min, max, n_samples):
    return cupy.random.uniform(min, max, n_samples, dtype=FLOAT_PRECISION)


def geomuniform(min, max, n_samples):
    min = cupy.log10(min)
    max = cupy.log10(max)
    exponents = cupy.random.uniform(min, max, n_samples)
    return cupy.power(10, exponents).astype(FLOAT_PRECISION)


def kroupa(min, max, n_samples):
    a = 2.3
    uniform_distribution = cupy.random.uniform(0, 1, n_samples)
    K = (1 - a) / (max ** (1 - a) - min ** (1 - a))
    Y = ((1 - a) / K * uniform_distribution + min ** (1 - a)) ** (1 / (1 - a))
    jj = cupy.logical_and((Y > min), (Y < max))
    return Y[jj].astype(FLOAT_PRECISION)


def geomspace(min, max, n_samples):
    # Cupy does not have geomspace implementation yet
    min = cupy.log10(min)
    max = cupy.log10(max)
    exponents = cupy.linspace(min, max, n_samples)
    return cupy.power(10, exponents).astype(FLOAT_PRECISION)


def linspace(min, max, n_samples):
    return cupy.linspace(min, max, n_samples, dtype=FLOAT_PRECISION)


def constant(min, max, n_samples):
    warnings.warn("Min value will be used as constant value, max is ignored.")
    return (cupy.zeros(n_samples) + min).astype(FLOAT_PRECISION)


def truncated_norm(max, min, n_samples):
    mean = (max + min) / 2
    sigma = (max + min) * 0.01
    remaining = n_samples
    out_arr = cupy.array([])
    while remaining > 0:
        arr = cupy.random.normal(mean, sigma, remaining).astype(numpy.float32)
        mask = cupy.logical_and((arr >= min), (arr <= max))
        out_arr = cupy.append(out_arr, arr[mask])
        remaining = len(arr[cupy.invert(mask)])
    return out_arr.astype(FLOAT_PRECISION)


def gaussian_3D(shape: list, radius=1):
    positions = cupy.random.multivariate_normal(
        [0, 0, 0],
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        size=shape,
    )
    normalized_positions = positions / cupy.abs(positions).max()
    rescaled_positions = normalized_positions * radius
    return rescaled_positions.astype(FLOAT_PRECISION)


def uniform_3D(shape: list, empty=False, radius=1):
    if empty:
        radius = 1
    else:
        cube_radius = cupy.random.uniform(0, 1, shape).astype(FLOAT_PRECISION)
        radius = cupy.power(cube_radius, 1 / 3)
    phi = cupy.random.uniform(0, 2 * cupy.pi, shape).astype(FLOAT_PRECISION)
    costheta = cupy.random.uniform(-1, 1, shape).astype(FLOAT_PRECISION)
    theta = cupy.arccos(costheta)

    x = radius * cupy.sin(theta) * cupy.cos(phi)
    y = radius * cupy.sin(theta) * cupy.sin(phi)
    z = radius * costheta

    positions = cupy.array([x, y, z], dtype=FLOAT_PRECISION)
    rescaled_positions = positions * radius
    reshaped_positions = cupy.einsum("ijk -> jki", rescaled_positions)

    return reshaped_positions.astype(FLOAT_PRECISION)


distributions = dict(
    UNIFORM=uniform,
    GEOMUNIFORM=geomuniform,
    KROUPA=kroupa,
    GEOMSPACE=geomspace,
    LINSPACE=linspace,
    TRUNCNORM=truncated_norm,
    CONSTANT=constant,
    GAUSS3D=gaussian_3D,
    UNIFORM3D=uniform_3D,
)
