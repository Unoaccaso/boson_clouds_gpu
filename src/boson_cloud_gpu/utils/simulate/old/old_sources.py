# ADDLICENSE

import sys
import os.path

PATH_TO_THIS = os.path.dirname(__file__)
PATH_TO_MASTER = PATH_TO_THIS + "/../../"
sys.path.append(PATH_TO_MASTER)

import cupy
import numpy
import warnings
import configparser

from ... import properties, distributions

PATH_TO_SETTINGS = PATH_TO_MASTER + "/config.ini"
config = configparser.ConfigParser()
config.read(PATH_TO_SETTINGS)

FLOAT_PRECISION = properties.FloatPrecision[
    config["numeric.precision"]["FloatPrecision"]
].value
INT_PRECISION = properties.IntPrecision[
    config["numeric.precision"]["IntPrecision"]
].value


class CustomDistributions:
    def __init__(
        self,
        n_samples: int,
        min: float,
        max: float,
    ) -> None:
        self._n_samples = n_samples
        self._min = min
        self._max = max

    def uniform(self):
        return cupy.random.uniform(
            self._min, self._max, self._n_samples, dtype=FLOAT_PRECISION
        )

    def geomuniform(self):
        min = cupy.log10(self._min)
        max = cupy.log10(self._max)
        exponents = cupy.random.uniform(min, max, self._n_samples)
        return cupy.power(10, exponents).astype(FLOAT_PRECISION)

    def kroupa(self):
        a = 2.3
        uniform_distribution = cupy.random.uniform(0, 1, self._n_samples)
        K = (1 - a) / (self._max ** (1 - a) - self._min ** (1 - a))
        Y = ((1 - a) / K * uniform_distribution + self._min ** (1 - a)) ** (1 / (1 - a))
        jj = cupy.logical_and((Y > self._min), (Y < self._max))
        return Y[jj].astype(FLOAT_PRECISION)

    def geomspace(self):
        # Cupy does not have geomspace implementation yet
        min = cupy.log10(self._min)
        max = cupy.log10(self._max)
        exponents = cupy.linspace(min, max, self._n_samples)
        return cupy.power(10, exponents).astype(FLOAT_PRECISION)

    def linspace(self):
        return cupy.linspace(
            self._min, self._max, self._n_samples, dtype=FLOAT_PRECISION
        )

    def truncated_norm(self):
        return truncated_norm(self._max, self._min, self._n_samples).astype(
            FLOAT_PRECISION
        )

    def constant(self):
        warnings.warn("Min value will be used as constant value, max is ignored.")
        return (cupy.zeros(self._n_samples) + self._min).astype(FLOAT_PRECISION)


class Masses(CustomDistributions):
    ...


class Spins(CustomDistributions):
    ...


class Ages(CustomDistributions):
    ...


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
    return out_arr
