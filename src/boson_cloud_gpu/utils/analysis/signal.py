# ADDLICENSE

from .. import generate

import sys
import os.path

PATH_TO_KERNELS = os.path.join(os.path.dirname(__file__), "../cuda_kernels/")
from functools import cached_property

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

    @cached_property
    def out_var(self):
        return cupy.empty((self.nrows, self.ncols), dtype=PRECISION)

    @property
    def df_dot(self):
        #! TODO: CONTROLLARE CHE QUESTO DFDOT VA CALCOLATO SULLE FREQUENZE AL DETECTOR
        df_dot_kernel = self.preprocessing_module.get_function("df_dot")
        _df_dot = self.dispatch_aritm_operation_kernel(
            df_dot_kernel,
            self.frequency_at_detector.astype(PRECISION),
        )

        return _df_dot

    @cached_property
    def alpha(self):
        alpha_kernel = self.preprocessing_module.get_function("alpha")
        _alpha = self.dispatch_aritm_operation_kernel(
            alpha_kernel,
            self.BH_mass.astype(PRECISION),
            self.boson_mass.astype(PRECISION),
        )

        return _alpha.astype(PRECISION)

    @cached_property
    def f_dot(self):
        f_dot_kernel = self.preprocessing_module.get_function("f_dot")
        _f_dot = self.dispatch_aritm_operation_kernel(
            f_dot_kernel,
            self.alpha.astype(PRECISION),
            self.boson_mass.astype(PRECISION),
        )

        return _f_dot

    @cached_property
    def tau_inst(self):
        tau_inst_kernel = self.preprocessing_module.get_function("tau_inst")
        _tau_inst = self.dispatch_aritm_operation_kernel(
            tau_inst_kernel,
            self.BH_mass.astype(PRECISION),
            self.spins.astype(PRECISION),
            self.alpha.astype(PRECISION),
        )

        return _tau_inst

    @cached_property
    def tau_gw(self):
        tau_gw_kernel = self.preprocessing_module.get_function("tau_gw")
        _tau_gw = self.dispatch_aritm_operation_kernel(
            tau_gw_kernel,
            self.BH_mass.astype(PRECISION),
            self.spins.astype(PRECISION),
            self.alpha.astype(PRECISION),
        )

        return _tau_gw

    @cached_property
    def chi_c(self):
        chi_c_kernel = self.preprocessing_module.get_function("chi_c")
        _chi_c = self.dispatch_aritm_operation_kernel(
            chi_c_kernel,
            self.alpha.astype(PRECISION),
        )

        return _chi_c

    @cached_property
    def frequency_at_detector(self):
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

    @cached_property
    def amplitude_at_detector(self):
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

    @cached_property
    def undetectable_values_mask(self):
        mask_kernel = self.preprocessing_module.get_function("build_mask")
        mask = self.dispatch_aritm_operation_kernel(
            mask_kernel,
            self.frequency_at_detector.astype(PRECISION),
            self.tau_gw.astype(PRECISION),
            self.tau_inst.astype(PRECISION),
            self.BH_ages_sec.astype(PRECISION),
            self.alpha.astype(PRECISION),
            self.spins.astype(PRECISION),
            self.chi_c.astype(PRECISION),
            self.f_dot.astype(PRECISION),
            self.df_dot.astype(PRECISION),
        )
        return mask

    @property
    def unmasked_frequencies(self):
        return self.frequency_at_detector

    @property
    def unmasked_amplitudes(self):
        return self.amplitude_at_detector

    @cached_property
    def frequency_amplitude(self):
        return self.dispatch_mask_kernel()

    def dispatch_mask_kernel(self):
        with open(PATH_TO_KERNELS + "/mask_kernel.cu", "r") as cuda_kernel:
            mask_kernel = cupy.RawKernel(cuda_kernel.read(), "mask_array")

        block_size = settings.CUDA["BLOCK_SIZE"]
        grid_size = (
            self.ncols // block_size[0] + 1,
            self.nrows // block_size[1] + 1,
        )
        masked_frequencies = cupy.empty_like(self.frequency_at_detector)
        masked_amplitudes = cupy.empty_like(self.amplitude_at_detector)
        mask_kernel(
            grid_size,
            block_size,
            (
                self.frequency_at_detector,
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
                masked_frequencies,
                masked_amplitudes,
            ),
        )

        return (masked_frequencies, masked_amplitudes)

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

    @cached_property
    def preprocessing_module(self):
        with open(PATH_TO_KERNELS + "/aritmetic_module.cu", "r") as cuda_module_file:
            cuda_module = cupy.RawModule(code=cuda_module_file.read())
        return cuda_module

    def plot(self):
        bh_ax, boson_ax = cupy.meshgrid(self.BH_mass, self.boson_mass)
