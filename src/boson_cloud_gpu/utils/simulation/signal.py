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

from utils.common import get_preprocessing_module

PATH_TO_SETTINGS = PATH_TO_MASTER + "/config.ini"
config = configparser.ConfigParser()
config.read(PATH_TO_SETTINGS)


from utils.common import FLOAT_PRECISION, INT_PRECISION


# global variables

BLOCK_SHAPE = (
    int(config["cuda"]["BlockSizeX"]),
    int(config["cuda"]["BlockSizeY"]),
)


def get_signals(BH_masses, BH_ages_yrs, BH_spins, distances, boson_masses):
    preprocessing_module = get_preprocessing_module("signal_kernel.cu")
    signal_kernel = preprocessing_module.get_function("get_signals")

    ncols = BH_masses.shape[0]
    nrows = boson_masses.shape[0]
    block_shape = BLOCK_SHAPE

    grid_shape = (
        ncols // block_shape[0] + 1,
        nrows // block_shape[1] + 1,
    )
    # In frequencies we add one row to store the value of minimum frequency per boson,
    # to calculate histograms faster.
    out_frequencies = cupy.ones((nrows, ncols), dtype=FLOAT_PRECISION)
    out_amplitudes = cupy.ones((nrows, ncols), dtype=FLOAT_PRECISION)
    signal_kernel(
        grid_shape,
        block_shape,
        (
            BH_masses,
            BH_ages_yrs,
            BH_spins,
            distances,
            boson_masses,
            nrows,
            ncols,
            out_frequencies,
            out_amplitudes,
        ),
    )

    return out_frequencies, out_amplitudes
