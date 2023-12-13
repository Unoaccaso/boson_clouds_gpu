# ADDLICENSE


import sys
import os.path

PATH_TO_THIS = os.path.dirname(__file__)
PATH_TO_KERNELS = PATH_TO_THIS
PATH_TO_MASTER = PATH_TO_THIS
sys.path.append(PATH_TO_MASTER)

# standard packages
import cupy

# user libraries
import configparser


PATH_TO_SETTINGS = PATH_TO_MASTER + "/config.ini"
config = configparser.ConfigParser()
config.read(PATH_TO_SETTINGS)


# global variables

BLOCK_SHAPE = (
    int(config["cuda"]["BlockSizeX"]),
    int(config["cuda"]["BlockSizeY"]),
)

import properties

FLOAT_PRECISION = properties.FloatPrecision[
    config["numeric.precision"]["FloatPrecision"]
].value
INT_PRECISION = properties.IntPrecision[
    config["numeric.precision"]["IntPrecision"]
].value


class SignalV2:
    def __init__(
        self,
        boson_mass,
        BH_mass,
        BH_ages_yrs,
        spins,
        distance,
        filter: bool = True,
    ) -> None:
        self._spins = spins
        self._filter = filter
        self._BH_mass = BH_mass
        self._boson_mass = boson_mass
        self._distance = distance
        self.BH_ages_sec = BH_ages_yrs * 86400 * 365
        self.nrows = len(self._boson_mass)
        self.ncols = len(self._BH_mass)

        self._alpha = self.alpha().copy()
        self._chi_c = self.chi_c().copy()
        self._f_dot = self.f_dot().copy()
        self._tau_inst = self.tau_inst().copy()
        self._tau_gw = self.tau_gw().copy()
        self.freq = self.frequency_at_detector().copy()
        self._df_dot = self.df_dot().copy()
        self.amp = self.amplitude_at_detector().copy()

    def df_dot(self):
        with open(PATH_TO_KERNELS + "/aritmetic_module.cu", "r") as cuda_module_file:
            preprocessing_module = cupy.RawModule(code=cuda_module_file.read())
        df_dot_kernel = preprocessing_module.get_function("df_dot")
        _df_dot = self.dispatch_aritm_operation_kernel(
            df_dot_kernel,
            self.freq.astype(FLOAT_PRECISION),
        )

        return _df_dot

    def alpha(self):
        with open(PATH_TO_KERNELS + "/aritmetic_module.cu", "r") as cuda_module_file:
            preprocessing_module = cupy.RawModule(code=cuda_module_file.read())
        alpha_kernel = preprocessing_module.get_function("alpha")
        _alpha = self.dispatch_aritm_operation_kernel(
            alpha_kernel,
            self._BH_mass.astype(FLOAT_PRECISION),
            self._boson_mass.astype(FLOAT_PRECISION),
        )

        return _alpha.astype(FLOAT_PRECISION)

    def f_dot(self):
        with open(PATH_TO_KERNELS + "/aritmetic_module.cu", "r") as cuda_module_file:
            preprocessing_module = cupy.RawModule(code=cuda_module_file.read())
        f_dot_kernel = preprocessing_module.get_function("f_dot")
        _f_dot = self.dispatch_aritm_operation_kernel(
            f_dot_kernel,
            self._alpha.astype(FLOAT_PRECISION),
            self._boson_mass.astype(FLOAT_PRECISION),
        )

        return _f_dot

    def df_dot(self):
        with open(PATH_TO_KERNELS + "/aritmetic_module.cu", "r") as cuda_module_file:
            preprocessing_module = cupy.RawModule(code=cuda_module_file.read())
        df_dot_kernel = preprocessing_module.get_function("df_dot")
        _df_dot = self.dispatch_aritm_operation_kernel(
            df_dot_kernel,
            self.freq.astype(FLOAT_PRECISION),
        )

        return _df_dot

    def tau_inst(self):
        with open(PATH_TO_KERNELS + "/aritmetic_module.cu", "r") as cuda_module_file:
            preprocessing_module = cupy.RawModule(code=cuda_module_file.read())
        tau_inst_kernel = preprocessing_module.get_function("tau_inst")
        _tau_inst = self.dispatch_aritm_operation_kernel(
            tau_inst_kernel,
            self._BH_mass.astype(FLOAT_PRECISION),
            self._spins.astype(FLOAT_PRECISION),
            self._alpha.astype(FLOAT_PRECISION),
        )

        return _tau_inst

    def tau_gw(self):
        with open(PATH_TO_KERNELS + "/aritmetic_module.cu", "r") as cuda_module_file:
            preprocessing_module = cupy.RawModule(code=cuda_module_file.read())
        tau_gw_kernel = preprocessing_module.get_function("tau_gw")
        _tau_gw = self.dispatch_aritm_operation_kernel(
            tau_gw_kernel,
            self._BH_mass.astype(FLOAT_PRECISION),
            self._spins.astype(FLOAT_PRECISION),
            self._alpha.astype(FLOAT_PRECISION),
        )

        return _tau_gw

    def chi_c(self):
        with open(PATH_TO_KERNELS + "/aritmetic_module.cu", "r") as cuda_module_file:
            preprocessing_module = cupy.RawModule(code=cuda_module_file.read())
        chi_c_kernel = preprocessing_module.get_function("chi_c")
        _chi_c = self.dispatch_aritm_operation_kernel(
            chi_c_kernel,
            self._alpha.astype(FLOAT_PRECISION),
        )

        return _chi_c

    def frequency_at_detector(self):
        with open(PATH_TO_KERNELS + "/aritmetic_module.cu", "r") as cuda_module_file:
            preprocessing_module = cupy.RawModule(code=cuda_module_file.read())
        frequency_kernel = preprocessing_module.get_function("frequency_at_detector")
        _frequency = self.dispatch_aritm_operation_kernel(
            frequency_kernel,
            self._BH_mass.astype(FLOAT_PRECISION),
            self._boson_mass.astype(FLOAT_PRECISION),
            self._f_dot.astype(FLOAT_PRECISION),
            self._tau_inst.astype(FLOAT_PRECISION),
            self.BH_ages_sec.astype(FLOAT_PRECISION),
        )

        return _frequency

    def amplitude_at_detector(self):
        with open(PATH_TO_KERNELS + "/aritmetic_module.cu", "r") as cuda_module_file:
            preprocessing_module = cupy.RawModule(code=cuda_module_file.read())
        amplitude_kernel = preprocessing_module.get_function("amplitude_at_detector")
        _amplitude = self.dispatch_aritm_operation_kernel(
            amplitude_kernel,
            self._BH_mass.astype(FLOAT_PRECISION),
            self._boson_mass.astype(FLOAT_PRECISION),
            self._spins.astype(FLOAT_PRECISION),
            self.BH_ages_sec.astype(FLOAT_PRECISION),
            self._distance.astype(FLOAT_PRECISION),
            self._alpha.astype(FLOAT_PRECISION),
            self._tau_inst.astype(FLOAT_PRECISION),
            self._tau_gw.astype(FLOAT_PRECISION),
            self._chi_c.astype(FLOAT_PRECISION),
        )

        return _amplitude

    def get_signals(self):
        with open(PATH_TO_KERNELS + "/mask_kernel.cu", "r") as cuda_kernel:
            mask_kernel = cupy.RawKernel(cuda_kernel.read(), "mask_array")

        block_size = BLOCK_SHAPE
        grid_size = (
            self.ncols // block_size[0] + 1,
            self.nrows // block_size[1] + 1,
        )
        masked_frequency = cupy.ones((self.nrows, self.ncols), dtype=FLOAT_PRECISION)
        masked_amplitude = cupy.ones((self.nrows, self.ncols), dtype=FLOAT_PRECISION)
        mask_kernel(
            grid_size,
            block_size,
            (
                self.freq,
                self.amp,
                self._tau_gw,
                self._tau_inst,
                self.BH_ages_sec,
                self._alpha,
                self._spins,
                self._chi_c,
                self._f_dot,
                self._df_dot,
                self.nrows,
                self.ncols,
                masked_frequency,
                masked_amplitude,
            ),
        )

        return (masked_frequency, masked_amplitude)

    def dispatch_aritm_operation_kernel(self, kernel, *args):
        block_size = BLOCK_SHAPE
        grid_size = (
            self.ncols // block_size[0] + 1,
            self.nrows // block_size[1] + 1,
        )

        # The output variable is created but the elements ar not initialized
        # one should be very sure that the kernel correctly populates the array.
        out_var = cupy.empty((self.nrows, self.ncols), dtype=FLOAT_PRECISION)
        kernel(grid_size, block_size, args + (self.nrows, self.ncols, out_var))

        return out_var

    def plot(self):
        bh_ax, boson_ax = cupy.meshgrid(self.BH_mass, self.boson_mass)
