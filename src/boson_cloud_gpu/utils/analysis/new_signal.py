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
    # DOCUMENT THIS
    preprocessing_module = get_preprocessing_module()

    BH_ages_sec = BH_ages_yrs * 365 * 86400

    alpha = compute_alpha(
        preprocessing_module,
        BH_masses,
        boson_masses,
    )
    f_dot = compute_f_dot(
        preprocessing_module,
        alpha,
        boson_masses,
    )
    tau_inst = compute_tau_inst(
        preprocessing_module,
        BH_masses,
        BH_spins,
        alpha,
    )
    tau_gw = compute_tau_inst(
        preprocessing_module,
        BH_masses,
        BH_spins,
        alpha,
    )
    chi_c = compute_chi_c(
        preprocessing_module,
        alpha,
    )
    frequency_at_detector = compute_frequency_at_detector(
        preprocessing_module,
        BH_masses,
        boson_masses,
        f_dot,
        tau_inst,
        BH_ages_sec,
    )
    amplitude_at_detector = compute_amplitude_at_detector(
        preprocessing_module,
        BH_masses,
        boson_masses,
        BH_spins,
        BH_ages_sec,
        distances,
        alpha,
        tau_inst,
        tau_gw,
        chi_c,
    )

    return (frequency_at_detector, amplitude_at_detector)


def get_preprocessing_module():
    # DOCUMENT THIS
    with open(PATH_TO_KERNELS + "/aritmetic_module.cu", "r") as cuda_module_file:
        cuda_module = cupy.RawModule(code=cuda_module_file.read())
    return cuda_module


def dispatch_aritm_operation_kernel(kernel, nrows, ncols, *args):
    # DOCUMENT THIS
    block_size = BLOCK_SHAPE

    grid_size = (
        ncols // block_size[0] + 1,
        nrows // block_size[1] + 1,
    )

    # The output variable is created but the elements ar not initialized
    # one should be very sure that the kernel correctly populates the array.
    out_var = cupy.ones((nrows, ncols), dtype=FLOAT_PRECISION)
    kernel(grid_size, block_size, args + (nrows, ncols, out_var))

    return out_var


def compute_alpha(preprocessing_module, BH_masses, boson_masses):
    # DOCUMENT THIS
    alpha_kernel = preprocessing_module.get_function("alpha")
    alpha = dispatch_aritm_operation_kernel(
        alpha_kernel,
        len(BH_masses),
        len(boson_masses),
        BH_masses.astype(FLOAT_PRECISION),
        boson_masses.astype(FLOAT_PRECISION),
    )

    return alpha.astype(FLOAT_PRECISION)


def compute_f_dot(preprocessing_module, alpha, boson_masses):
    # DOCUMENT THIS
    f_dot_kernel = preprocessing_module.get_function("f_dot")
    f_dot = dispatch_aritm_operation_kernel(
        f_dot_kernel,
        alpha.shape[0],
        alpha.shape[1],
        alpha.astype(FLOAT_PRECISION),
        boson_masses.astype(FLOAT_PRECISION),
    )

    return f_dot


def compute_tau_inst(preprocessing_module, BH_masses, BH_spins, alpha):
    # DOCUMENT THIS
    tau_inst_kernel = preprocessing_module.get_function("tau_inst")
    tau_inst = dispatch_aritm_operation_kernel(
        tau_inst_kernel,
        alpha.shape[0],
        alpha.shape[1],
        BH_masses.astype(FLOAT_PRECISION),
        BH_spins.astype(FLOAT_PRECISION),
        alpha.astype(FLOAT_PRECISION),
    )

    return tau_inst


def compute_tau_gw(preprocessing_module, BH_masses, BH_spins, alpha):
    # DOCUMENT THIS
    tau_gw_kernel = preprocessing_module.get_function("tau_gw")
    tau_gw = dispatch_aritm_operation_kernel(
        tau_gw_kernel,
        alpha.shape[0],
        alpha.shape[1],
        BH_masses.astype(FLOAT_PRECISION),
        BH_spins.astype(FLOAT_PRECISION),
        alpha.astype(FLOAT_PRECISION),
    )

    return tau_gw


def compute_chi_c(preprocessing_module, alpha):
    # DOCUMENT THIS
    chi_c_kernel = preprocessing_module.get_function("chi_c")
    _chi_c = dispatch_aritm_operation_kernel(
        chi_c_kernel,
        alpha.shape[0],
        alpha.shape[1],
        alpha.astype(FLOAT_PRECISION),
    )

    return _chi_c


def compute_frequency_at_detector(
    preprocessing_module,
    BH_masses,
    boson_masses,
    f_dot,
    tau_inst,
    BH_ages_sec,
):
    # DOCUMENT THIS
    frequency_kernel = preprocessing_module.get_function("frequency_at_detector")
    _frequency = dispatch_aritm_operation_kernel(
        frequency_kernel,
        len(BH_masses),
        len(boson_masses),
        BH_masses.astype(FLOAT_PRECISION),
        boson_masses.astype(FLOAT_PRECISION),
        f_dot.astype(FLOAT_PRECISION),
        tau_inst.astype(FLOAT_PRECISION),
        BH_ages_sec.astype(FLOAT_PRECISION),
    )

    return _frequency


def compute_amplitude_at_detector(
    preprocessing_module,
    BH_masses,
    boson_masses,
    BH_spins,
    BH_ages_sec,
    distances,
    alpha,
    tau_inst,
    tau_gw,
    chi_c,
):
    # DOCUMENT THIS
    amplitude_kernel = preprocessing_module.get_function("amplitude_at_detector")
    _amplitude = dispatch_aritm_operation_kernel(
        amplitude_kernel,
        len(BH_masses),
        len(boson_masses),
        BH_masses.astype(FLOAT_PRECISION),
        boson_masses.astype(FLOAT_PRECISION),
        BH_spins.astype(FLOAT_PRECISION),
        BH_ages_sec.astype(FLOAT_PRECISION),
        distances.astype(FLOAT_PRECISION),
        alpha.astype(FLOAT_PRECISION),
        tau_inst.astype(FLOAT_PRECISION),
        tau_gw.astype(FLOAT_PRECISION),
        chi_c.astype(FLOAT_PRECISION),
    )

    return _amplitude
