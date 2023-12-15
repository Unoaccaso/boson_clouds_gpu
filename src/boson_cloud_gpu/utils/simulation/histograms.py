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

import cupy
import numpy

import sys
import os.path

import matplotlib.pyplot as plt

PATH_TO_THIS = os.path.dirname(__file__)
PATH_TO_MASTER = PATH_TO_THIS + "/../../"
PATH_TO_KERNELS = PATH_TO_MASTER + "/utils/cuda_kernels/"
sys.path.append(PATH_TO_MASTER)

import cupy
import configparser

from utils.common import get_preprocessing_module

PATH_TO_SETTINGS = PATH_TO_MASTER + "/config.ini"
config = configparser.ConfigParser()
config.read(PATH_TO_SETTINGS)


BLOCK_SHAPE = (
    int(config["cuda"]["BlockSizeX"]),
    int(config["cuda"]["BlockSizeY"]),
)

from utils.common import FLOAT_PRECISION, INT_PRECISION


def cupy_histograms(frequencies_gpu, amplitudes, t_fft, band_size: int):
    mean_frequencies = cupy.nanmean(frequencies_gpu, axis=1).astype(FLOAT_PRECISION)

    nbins = INT_PRECISION(band_size * t_fft)
    nrows = INT_PRECISION(frequencies_gpu.shape[0])

    bins_gpu = get_bins(
        mean_frequencies,
        band_size,
        t_fft,
        nbins,
        nrows,
    )
    frequencies_cpu, bins_cpu = frequencies_gpu.get(), bins_gpu.get()

    counts = histogram_along_axis_cpu(frequencies_cpu, bins_cpu)
    return counts, bins_cpu[:, :-1]


def cuda_histograms(frequencies_gpu, amplitudes_gpu, t_fft, band_size):
    mean_frequencies = cupy.nanmean(frequencies_gpu, axis=1).astype(FLOAT_PRECISION)

    nbins = INT_PRECISION(band_size * t_fft)
    nrows = INT_PRECISION(frequencies_gpu.shape[0])
    ncols = INT_PRECISION(frequencies_gpu.shape[1])

    bins = get_bins(
        mean_frequencies,
        band_size,
        t_fft,
        nbins,
        nrows,
    )
    counts = get_counts(
        frequencies_gpu,
        amplitudes_gpu,
        mean_frequencies,
        band_size,
        t_fft,
        nbins,
        ncols,
        nrows,
    )

    return counts, bins


def get_counts(
    frequencies, amplitudes, mean_frequencies, band_size, t_fft, nbins, ncols, nrows
):
    preprocessing_module = get_preprocessing_module("histogram_kernel.cu")
    counts = cupy.zeros((nrows, nbins), dtype=FLOAT_PRECISION)
    block_shape = BLOCK_SHAPE
    grid_shape = (
        ncols // block_shape[0] + 1,
        nrows // block_shape[1] + 1,
    )
    kernel = preprocessing_module.get_function("make_histograms")
    kernel(
        grid_shape,
        block_shape,
        (
            frequencies,
            amplitudes,
            mean_frequencies,
            INT_PRECISION(band_size),
            FLOAT_PRECISION(t_fft),
            nbins,
            ncols,
            nrows,
            counts,
        ),
    )
    return counts


def get_bins(mean_frequencies, band_size, t_fft, nbins, nrows):
    preprocessing_module = get_preprocessing_module("histogram_kernel.cu")
    bins = cupy.ones((nrows, nbins), dtype=FLOAT_PRECISION)
    block_shape = BLOCK_SHAPE
    grid_shape = (
        INT_PRECISION(nbins // block_shape[0] + 1),
        INT_PRECISION(nrows // block_shape[1] + 1),
    )
    kernel = preprocessing_module.get_function("make_bins")
    kernel(
        grid_shape,
        block_shape,
        (
            mean_frequencies,
            INT_PRECISION(band_size),
            FLOAT_PRECISION(t_fft),
            nrows,
            nbins,
            bins,
        ),
    )
    return bins


def histogram_along_axis_cpu(frequencies, bins):
    counts = numpy.ones((bins.shape[0], bins.shape[1] - 1))

    for row, frequency_arr in enumerate(frequencies):
        counts[row] = numpy.histogram(
            frequency_arr[~numpy.isnan(frequency_arr)], bins[row]
        )[0]

    return counts
