# ADDLICENSE
# %%

import cupy
import numpy
from numba import cuda

import warnings


class Masses:
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
            self._min, self._max, self._n_samples, dtype=cupy.float32
        )

    def kroupa(self):
        a = 2.3
        uniform_distribution = cupy.random.uniform(0, 1, self._n_samples)
        K = (1 - a) / (self._max ** (1 - a) - self._min ** (1 - a))
        Y = ((1 - a) / K * uniform_distribution + self._min ** (1 - a)) ** (1 / (1 - a))
        jj = cupy.logical_and((Y > self._min), (Y < self._max))
        return Y[jj].astype(cupy.float32)

    def logspace(self):
        return cupy.logspace(self._min, self._max, self._n_samples, dtype=cupy.float32)

    def linspace(self):
        return cupy.linspace(self._min, self._max, self._n_samples, dtype=cupy.float32)


class Spins:
    def __init__(
        self,
        n_samples: int,
        min: float,
        max: float = 1,
    ) -> None:
        self._n_samples = n_samples
        self._min = min
        self._max = max

    def uniform(self):
        return cupy.random.uniform(self._min, self._max, self._n_samples, cupy.float32)

    def truncated_norm(self):
        return truncated_norm(self._max, self._min, self._n_samples)

    def constant(self):
        warnings.warn("Min value will be used as spin value, max is ignored.")
        return cupy.zeros(self._n_samples, dtype=cupy.float32) + self._min


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
