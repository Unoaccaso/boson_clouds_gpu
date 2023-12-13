# ADDLICENSE


import sys
import os.path

PATH_TO_THIS = os.path.dirname(__file__)
PATH_TO_KERNELS = PATH_TO_THIS + "/../cuda_kernels/"
PATH_TO_MASTER = PATH_TO_THIS + "/../../"
sys.path.append(PATH_TO_MASTER)

# standard packages
import cupy

# user libraries
import configparser
from . import properties

PATH_TO_SETTINGS = PATH_TO_MASTER + "/config.ini"
config = configparser.ConfigParser()
config.read(PATH_TO_SETTINGS)


# global variables

BLOCK_SHAPE = (
    int(config["cuda"]["BlockSizeX"]),
    int(config["cuda"]["BlockSizeY"]),
)

FLOAT_PRECISION = properties.FloatPrecision[
    config["numeric.precision"]["FloatPrecision"]
].value
INT_PRECISION = properties.IntPrecision[
    config["numeric.precision"]["IntPrecision"]
].value


def generate_signals(BH_masses, BH_ages_yrs, BH_spins, distances, boson_masses):
    preprocessing_module = get_preprocessing_module("signal_kernel.cu")
    signal_kernel = preprocessing_module.get_function("get_signals")

    ncols = len(BH_masses)
    nrows = len(boson_masses)
    block_size = BLOCK_SHAPE

    grid_size = (
        ncols // block_size[0] + 1,
        nrows // block_size[1] + 1,
    )
    out_frequencies = cupy.ones((ncols, nrows), dtype=FLOAT_PRECISION)
    out_amplitudes = cupy.ones((ncols, nrows), dtype=FLOAT_PRECISION)
    signal_kernel(
        grid_size,
        block_size,
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


def calculate_distances(positions):
    # DOCUMENT THIS
    with open(PATH_TO_KERNELS + "/distance_kernel.cu", "r") as cuda_kernel:
        distance_kernel = cupy.RawKernel(cuda_kernel.read(), "distance")
    n_positions = int(positions.shape[0])
    distances = cupy.empty(n_positions, dtype=FLOAT_PRECISION)
    block_shape = (int(config["cuda"]["BlockSizeX"]),)
    grid_shape = (n_positions // block_shape[0] + 1,)
    distance_kernel(grid_shape, block_shape, (positions, n_positions, distances))

    return distances


def get_preprocessing_module(module: str):
    # DOCUMENT THIS
    with open(PATH_TO_KERNELS + "/" + module, "r") as cuda_module_file:
        cuda_module = cupy.RawModule(code=cuda_module_file.read())
    return cuda_module
