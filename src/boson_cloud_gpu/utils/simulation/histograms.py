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


import sys
import os.path

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


def get_bins(median_frequencies, band_size, t_fft, nbins, nrows):
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
            median_frequencies,
            INT_PRECISION(band_size),
            FLOAT_PRECISION(t_fft),
            nrows,
            nbins,
            bins,
        ),
    )

    return bins


def cupy_histograms(frequencies, amplitudes, t_fft, band_size: int):
    median_frequencies = cupy.nanmedian(frequencies, axis=1)

    nbins = INT_PRECISION(band_size * t_fft)
    nrows = INT_PRECISION(frequencies.shape[0])
    bins = get_bins(
        median_frequencies,
        band_size,
        t_fft,
        nbins,
        nrows,
    )


def cuda_histograms():
    ...
