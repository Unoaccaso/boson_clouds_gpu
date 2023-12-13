# ADDLICENSE

from .. import generate

import sys
import os.path

from functools import cached_property

PATH_TO_KERNELS = os.path.join(os.path.dirname(__file__), "../cuda_kernels/")

sys.path.append("../../")
import settings

import cupy
from cupy.typing import NDArray


PRECISION = settings.GENERAL["PRECISION"]


def dispatch_distance_kernel(positions: NDArray):
    # Computing distances
    with open(PATH_TO_KERNELS + "/distance_kernel.cu", "r") as cuda_kernel:
        distance_kernel = cupy.RawKernel(cuda_kernel.read(), "distance")
    n_positions = int(positions.shape[0])
    distances = cupy.empty(n_positions, dtype=PRECISION)
    block_size = (1, settings.CUDA["BLOCK_SIZE"][1])
    grid_size = (1, n_positions // int(block_size[1]) + 1)
    distance_kernel(grid_size, block_size, (positions, n_positions, distances))

    return distances


class Signal:
    def __init__(
        self,
        boson_mass: NDArray,
        BH_mass: NDArray,
        BH_ages_yrs: NDArray,
        spins: NDArray,
        distance: NDArray,
        filter: bool = True,
    ) -> None:
        self._spins = spins
        self._filter = filter
        self._BH_mass = BH_mass
        self._boson_mass = boson_mass
        self._distance = distance
        self.BH_ages_sec = BH_ages_yrs * 86400 * 365
        self.nrows = len(self.boson_mass)
        self.ncols = len(self.BH_mass)

        self.preprocessing_module = self.get_preprocessing_module()
        self.alpha = self.compute_alpha()
        print(self.alpha)
        self.f_dot = self.compute_f_dot()
        self.df_dot = self.compute_df_dot()
        self.tau_inst = self.compute_tau_inst()
        self.tau_gw = self.compute_tau_gw()
        self.frequency_at_detector = self.compute_frequency_at_detector()
        self.chi_c = self.compute_chi_c()
        self.amplitude_at_detector = self.compute_amplitude_at_detector()

    @property
    def distance(self):
        return self._distance

    @property
    def boson_mass(self):
        return self._boson_mass

    @property
    def BH_mass(self):
        return self._BH_mass

    @property
    def spins(self):
        return self._spins

    def get_preprocessing_module(self):
        with open(PATH_TO_KERNELS + "/aritmetic_module.cu", "r") as cuda_module_file:
            cuda_module = cupy.RawModule(code=cuda_module_file.read())
        return cuda_module

    def out_var(self):
        return cupy.empty((self.nrows, self.ncols), dtype=PRECISION)

    def df_dot(self):
        df_dot_kernel = self.preprocessing_module.get_function("df_dot")
        _df_dot = self.dispatch_aritm_operation_kernel(
            df_dot_kernel,
            self.frequency_at_detector.astype(PRECISION),
        )

        return _df_dot

    def compute_alpha(self):
        alpha_kernel = self.preprocessing_module.get_function("alpha")
        _alpha = self.dispatch_aritm_operation_kernel(
            alpha_kernel,
            self.BH_mass.astype(PRECISION),
            self.boson_mass.astype(PRECISION),
        )

        return _alpha.astype(PRECISION)

    def compute_f_dot(self):
        f_dot_kernel = self.preprocessing_module.get_function("f_dot")
        _f_dot = self.dispatch_aritm_operation_kernel(
            f_dot_kernel,
            self.alpha.astype(PRECISION),
            self.boson_mass.astype(PRECISION),
        )

        return _f_dot

    def compute_df_dot(self):
        #! TODO: CONTROLLARE CHE QUESTO DFDOT VA CALCOLATO SULLE FREQUENZE AL DETECTOR
        df_dot_kernel = self.preprocessing_module.get_function("df_dot")
        _df_dot = self.dispatch_aritm_operation_kernel(
            df_dot_kernel,
            self.frequency_at_detector.astype(PRECISION),
        )

        return _df_dot

    def compute_tau_inst(self):
        tau_inst_kernel = self.preprocessing_module.get_function("tau_inst")
        _tau_inst = self.dispatch_aritm_operation_kernel(
            tau_inst_kernel,
            self.BH_mass.astype(PRECISION),
            self.spins.astype(PRECISION),
            self.alpha.astype(PRECISION),
        )

        return _tau_inst

    def compute_tau_gw(self):
        tau_gw_kernel = self.preprocessing_module.get_function("tau_gw")
        _tau_gw = self.dispatch_aritm_operation_kernel(
            tau_gw_kernel,
            self.BH_mass.astype(PRECISION),
            self.spins.astype(PRECISION),
            self.alpha.astype(PRECISION),
        )

        return _tau_gw

    def compute_chi_c(self):
        chi_c_kernel = self.preprocessing_module.get_function("chi_c")
        _chi_c = self.dispatch_aritm_operation_kernel(
            chi_c_kernel,
            self.alpha.astype(PRECISION),
        )

        return _chi_c

    def compute_frequency_at_detector(self):
        frequency_kernel = self.preprocessing_module.get_function(
            "frequency_at_detector"
        )
        _frequency = self.dispatch_aritm_operation_kernel(
            frequency_kernel,
            self.BH_mass.astype(PRECISION),
            self.boson_mass.astype(PRECISION),
            self.f_dot.astype(PRECISION),
            self.tau_inst.astype(PRECISION),
            self.BH_ages_sec.astype(PRECISION),
        )

        return _frequency

    def compute_amplitude_at_detector(self):
        amplitude_kernel = self.preprocessing_module.get_function(
            "amplitude_at_detector"
        )
        _amplitude = self.dispatch_aritm_operation_kernel(
            amplitude_kernel,
            self.BH_mass.astype(PRECISION),
            self.boson_mass.astype(PRECISION),
            self.spins.astype(PRECISION),
            self.BH_ages_sec.astype(PRECISION),
            self.distance.astype(PRECISION),
            self.alpha.astype(PRECISION),
            self.tau_inst.astype(PRECISION),
            self.tau_gw.astype(PRECISION),
            self.chi_c.astype(PRECISION),
        )

        return _amplitude

    @property
    def unmasked_frequencies(self):
        return self.frequency_at_detector

    @property
    def unmasked_amplitudes(self):
        return self.amplitude_at_detector

    def get_signals(self):
        with open(PATH_TO_KERNELS + "/mask_kernel.cu", "r") as cuda_kernel:
            mask_kernel = cupy.RawKernel(cuda_kernel.read(), "mask_array")

        block_size = settings.CUDA["BLOCK_SIZE"]
        grid_size = (
            self.ncols // block_size[0] + 1,
            self.nrows // block_size[1] + 1,
        )
        masked_frequency = self.frequency_at_detector.copy()
        masked_amplitude = self.amplitude_at_detector.compy()
        mask_kernel(
            grid_size,
            block_size,
            (
                self.frequency_at_detector,
                self.amplitude_at_detector,
                self.tau_gw,
                self.tau_inst,
                self.BH_ages_sec,
                self.alpha,
                self.spins,
                self.chi_c,
                self.f_dot,
                self.df_dot,
                self.nrows,
                self.ncols,
            ),
        )

        return (masked_frequency, masked_amplitude)

    def dispatch_aritm_operation_kernel(self, kernel, *args):
        block_size = settings.CUDA["BLOCK_SIZE"]

        grid_size = (
            self.ncols // block_size[0] + 1,
            self.nrows // block_size[1] + 1,
        )

        # The output variable is created but the elements ar not initialized
        # one should be very sure that the kernel correctly populates the array.
        kernel(grid_size, block_size, args + (self.nrows, self.ncols, self.out_var))

        return self.out_var

    def plot(self):
        bh_ax, boson_ax = cupy.meshgrid(self.BH_mass, self.boson_mass)
