# ADDLICENSE

from .. import generate

import sys
from functools import cached_property

sys.path.append("../../")
import settings

import cupy
from cupy.typing import NDArray


PRECISION = settings.GENERAL["PRECISION"]


_preprocessing_module = r"""
extern "C"{
    #define G 6.6743e-11
    #define c 299792458.0
    #define hbar 1.0545718176461565e-34
    #define m_sun 1.988409870698051e+30
    #define om0 7.27220521664304e-05
    #define r0 5.5e6
    #define onev 1.60217653e-19
    #define f_int 1e30
    #define duty 0.7
    #define t_obs 365 * 86400 * duty
    #define NAN 0.0/0.0

    __global__ void alpha(
        const float* bh_mass,
        const float* boson_mass,
        const int nrows,
        const int ncols,
        float* _alpha
    ){  
        
        int x_abs = threadIdx.x + blockDim.x * blockIdx.x;
        int y_abs = threadIdx.y + blockDim.y * blockIdx.y;
        
        if ((x_abs < ncols) && (y_abs < nrows)){
            _alpha[x_abs + ncols * y_abs] = G / (c * c * c * hbar) * 2e30\
                * bh_mass[x_abs] * boson_mass[y_abs] * onev;
        }
    }

    __global__ void f_dot(
        const float* alpha,
        const float* boson_mass,
        const int nrows,
        const int ncols,
        float* f_dot
    ){  
        
        int x_abs = threadIdx.x + blockDim.x * blockIdx.x;
        int y_abs = threadIdx.y + blockDim.y * blockIdx.y;
        
        if ((x_abs < ncols) && (y_abs < nrows)){
        
            // faster power to 17 ~ half the time
            float alpha_17 = alpha[x_abs + ncols * y_abs] / 0.1;
            int k;
            for (k = 0 ; k < 4 ; k++)
            {
                alpha_17 *= alpha_17;
            }
            alpha_17 *= alpha[x_abs + ncols * y_abs] / 0.1;

            float A = 1e-10 * ( 1e17 / f_int );
            float A_4 = A * A * A * A;

            f_dot[x_abs + ncols * y_abs] = 7e-15 + A_4 * alpha_17 * \
                (boson_mass[y_abs]/1.0e-12) * (boson_mass[y_abs]/1.0e-12);
        }        
    }

    __global__ void df_dot(
        const float* frequency,
        const int nrows,
        const int ncols,
        float* df_dot
    ){  
        
        int x_abs = threadIdx.x + blockDim.x * blockIdx.x;
        int y_abs = threadIdx.y + blockDim.y * blockIdx.y;
        
        if ((x_abs < ncols) && (y_abs < nrows)){
            float ceiled = ceil(frequency[x_abs + ncols * y_abs] / 10.) * 10.;
            float dfr = om0 * sqrt(2 * ceiled * r0 / c);
            df_dot[x_abs + ncols * y_abs] = dfr / (2 * t_obs / duty);
        }        
    }

    __global__ void tau_inst(
        const float* bh_mass,
        const float* spin,
        const float* alpha,
        const int nrows,
        const int ncols,
        float* tau_inst
    ){  
        
        int x_abs = threadIdx.x + blockDim.x * blockIdx.x;
        int y_abs = threadIdx.y + blockDim.y * blockIdx.y;
        
        if ((x_abs < ncols) && (y_abs < nrows)){
        
            // faster power to 9 ~ half the time
            float alpha_9 = alpha[x_abs + ncols * y_abs] / 0.1;
            int k;
            for (k = 0 ; k < 3 ; k++)
            {
                alpha_9 *= alpha_9;
            }
            alpha_9 *= alpha[x_abs + ncols * y_abs] / 0.1;
        
            tau_inst[x_abs + ncols * y_abs] = 27 * 86400 / 10 * bh_mass[x_abs]\
                / alpha_9 / spin[x_abs];
        }
    }

    __global__ void tau_gw(
        const float* bh_mass,
        const float* spin,
        const float* alpha,
        const int nrows,
        const int ncols,
        float* tau_gw
    ){  
        
        int x_abs = threadIdx.x + blockDim.x * blockIdx.x;
        int y_abs = threadIdx.y + blockDim.y * blockIdx.y;
        
        if ((x_abs < ncols) && (y_abs < nrows)){
        
            // faster power to 15 ~ half the time
            float alpha_15 = alpha[x_abs + ncols * y_abs] / 0.1;
            int k;
            for (k = 0 ; k < 4 ; k++)
            {
                alpha_15 *= alpha_15;
            }
            alpha_15 /= alpha[x_abs + ncols * y_abs] / 0.1;

            tau_gw[x_abs + ncols * y_abs] = 6.5e4 * 365 * 86400 * bh_mass[x_abs]\
                / 10 / alpha_15 / spin[x_abs];
        }
    }

    __global__ void chi_c(
        const float* alpha,
        const int nrows,
        const int ncols,
        float* chi_c
    ){  
        
        int x_abs = threadIdx.x + blockDim.x * blockIdx.x;
        int y_abs = threadIdx.y + blockDim.y * blockIdx.y;
        
        if ((x_abs < ncols) && (y_abs < nrows)){
            float _alpha = alpha[x_abs + ncols * y_abs];
            chi_c[x_abs + ncols * y_abs] = 4 * _alpha\
                / ( 1 +4 * _alpha * _alpha);
        }        
    }

    __global__ void frequency_at_detector(
        const float* bh_mass,
        const float* boson_mass,
        const float* f_dot,
        const float* tau_inst,
        const float* bh_age_sec,
        const int nrows,
        const int ncols,
        float* frequency
    ){  
        
        int x_abs = threadIdx.x + blockDim.x * blockIdx.x;
        int y_abs = threadIdx.y + blockDim.y * blockIdx.y;
        
        if ((x_abs < ncols) && (y_abs < nrows)){
            float emitted_freq = 483 * (boson_mass[y_abs] / 1.0e-12)\
                *(1 - 0.0056 / 8 * (bh_mass[x_abs] / 10) *(boson_mass[y_abs] / 1.e-12)\
                    * (bh_mass[x_abs] / 10) *(boson_mass[y_abs] / 1.e-12));

            frequency[x_abs + ncols * y_abs] = emitted_freq + f_dot[x_abs + ncols * y_abs]\
                 * (bh_age_sec[x_abs] - tau_inst[x_abs + ncols * y_abs]);
        }
    }

    __global__ void amplitude_at_detector(
        const float* bh_mass,
        const float* boson_mass,
        const float* spin,
        const float* bh_age_sec,
        const float* distance,
        const float* alpha,
        const float* tau_inst,
        const float* tau_gw,
        const float* chi_c,
        const int nrows,
        const int ncols,
        float* amplitude
    ){  
        
        int x_abs = threadIdx.x + blockDim.x * blockIdx.x;
        int y_abs = threadIdx.y + blockDim.y * blockIdx.y;
        
        if ((x_abs < ncols) && (y_abs < nrows)){
        
            // faster power to 7 ~ half the time
            float alpha_7 = alpha[x_abs + ncols * y_abs] / 0.1;
            int k;
            for (k = 0 ; k < 3 ; k++)
            {
                alpha_7 *= alpha_7;
            }
            alpha_7 /= alpha[x_abs + ncols * y_abs] / 0.1;

            float timefactor = ( 1 + (bh_age_sec[x_abs] -\
                  tau_inst[x_abs + ncols * y_abs]) / tau_gw[x_abs + ncols * y_abs]);
                
            float amp_at_1kpc = 3.0e-24 / 10 * bh_mass[x_abs] *\
                alpha_7 * (spin[x_abs] - chi_c[x_abs + ncols * y_abs]) / 0.5;

            amplitude[x_abs + ncols * y_abs] = amp_at_1kpc / (timefactor * distance[x_abs]);   
        }          
    }

    __global__ void build_mask(
        const float* frequency,
        const float* tau_gw,
        const float* tau_inst,
        const float* bh_age_sec,
        const float* alpha,
        const float* spin,
        const float* chi_c,
        const float* f_dot,
        const float* df_dot,
        const int nrows,
        const int ncols,    
        float* mask)
    {   
        
        int x_abs = threadIdx.x + blockDim.x * blockIdx.x;
        int y_abs = threadIdx.y + blockDim.y * blockIdx.y;
        
        if ((x_abs < ncols) && (y_abs < nrows))
        {
            if (
                (tau_gw[x_abs + ncols * y_abs] < 10 * bh_age_sec[x_abs]) \
                || (10 * tau_inst[x_abs + ncols * y_abs] > bh_age_sec[x_abs])\
                || (alpha[x_abs + ncols * y_abs] > 0.1) \
                || (frequency[x_abs + ncols * y_abs] < 20.) \
                || (frequency[x_abs + ncols * y_abs] > 2048.) \
                || (10 * tau_inst[x_abs + ncols * y_abs] > tau_gw[x_abs + ncols * y_abs]) \
                || (spin[x_abs] < chi_c[x_abs + ncols * y_abs]) \
                || (df_dot[x_abs + ncols * y_abs] < f_dot[x_abs + ncols * y_abs])
            )
            {
                mask[x_abs + ncols * y_abs] = 0;
            }
        }
    }

    __global__ void apply_mask(
        const float* arr,
        const float* mask,
        const int nrows,
        const int ncols,
        float* out
    )
    {   
        
        int x_abs = threadIdx.x + blockDim.x * blockIdx.x;
        int y_abs = threadIdx.y + blockDim.y * blockIdx.y;
        
        if ((x_abs < ncols) && (y_abs < nrows))
        {
            out[x_abs + ncols * y_abs] = arr[x_abs + ncols * y_abs] * mask[x_abs + ncols * y_abs];
        }
    }
}
"""

preprocessing_module = cupy.RawModule(code=_preprocessing_module)


def distance(positions: NDArray):
    # Computing distances
    distances = cupy.sqrt(
        positions[:, 0] * positions[:, 0]
        + positions[:, 1] * positions[:, 1]
        + positions[:, 2] * positions[:, 2]
    )
    del positions

    return distances


def dispatch_kernel(kernel, nrows: int, ncols: int, *args):
    block_size = settings.CUDA["BLOCK_SIZE"]

    grid_size = (
        int(cupy.ceil(ncols / block_size[0])),
        int(cupy.ceil(nrows / block_size[1])),
    )

    # This runs faster on teslak20 (empirical result)
    _1D_grid_size = (
        int(
            cupy.ceil(
                ncols / (block_size[0] * block_size[1]),
            )
        ),
    )

    out_var = cupy.ones((nrows, ncols), dtype=PRECISION)
    kernel(_1D_grid_size, block_size, args + (nrows, ncols, out_var))

    return out_var


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
        self.n_rows = len(self.boson_mass)
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

    @property
    def df_dot(self):
        #! TODO: CONTROLLARE CHE QUESTO DFDOT VA CALCOLATO SULLE FREQUENZE AL DETECTOR
        df_dot_kernel = preprocessing_module.get_function("df_dot")
        _df_dot = dispatch_kernel(
            df_dot_kernel,
            self.n_rows,
            self.ncols,
            self.frequency_at_detector.astype(PRECISION),
        )

        return _df_dot

    @cached_property
    def BH_mass_2(self):
        return cupy.power(self._BH_mass, 2)

    @cached_property
    def boson_mass_2(self):
        return cupy.power(self._boson_mass, 2)

    @cached_property
    def alpha(self):
        alpha_kernel = preprocessing_module.get_function("alpha")
        _alpha = dispatch_kernel(
            alpha_kernel,
            self.n_rows,
            self.ncols,
            self.BH_mass.astype(PRECISION),
            self.boson_mass.astype(PRECISION),
        )

        return _alpha.astype(PRECISION)

    @cached_property
    def f_dot(self):
        f_dot_kernel = preprocessing_module.get_function("f_dot")
        _f_dot = dispatch_kernel(
            f_dot_kernel,
            self.n_rows,
            self.ncols,
            self.alpha.astype(PRECISION),
            self.boson_mass.astype(PRECISION),
        )

        return _f_dot

    @cached_property
    def tau_inst(self):
        tau_inst_kernel = preprocessing_module.get_function("tau_inst")
        _tau_inst = dispatch_kernel(
            tau_inst_kernel,
            self.n_rows,
            self.ncols,
            self.BH_mass.astype(PRECISION),
            self.spins.astype(PRECISION),
            self.alpha.astype(PRECISION),
        )

        return _tau_inst

    @cached_property
    def tau_gw(self):
        tau_gw_kernel = preprocessing_module.get_function("tau_gw")
        _tau_gw = dispatch_kernel(
            tau_gw_kernel,
            self.n_rows,
            self.ncols,
            self.BH_mass.astype(PRECISION),
            self.spins.astype(PRECISION),
            self.alpha.astype(PRECISION),
        )

        return _tau_gw

    @cached_property
    def chi_c(self):
        chi_c_kernel = preprocessing_module.get_function("chi_c")
        _chi_c = dispatch_kernel(
            chi_c_kernel,
            self.n_rows,
            self.ncols,
            self.alpha.astype(PRECISION),
        )

        return _chi_c

    @cached_property
    def frequency_at_detector(self):
        frequency_kernel = preprocessing_module.get_function("frequency_at_detector")
        _frequency = dispatch_kernel(
            frequency_kernel,
            self.n_rows,
            self.ncols,
            self.BH_mass.astype(PRECISION),
            self.boson_mass.astype(PRECISION),
            self.f_dot.astype(PRECISION),
            self.tau_inst.astype(PRECISION),
            self.BH_ages_sec.astype(PRECISION),
        )

        return _frequency

    @cached_property
    def amplitude_at_detector(self):
        amplitude_kernel = preprocessing_module.get_function("amplitude_at_detector")
        _amplitude = dispatch_kernel(
            amplitude_kernel,
            self.n_rows,
            self.ncols,
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
        mask_kernel = preprocessing_module.get_function("build_mask")
        mask = dispatch_kernel(
            mask_kernel,
            self.n_rows,
            self.ncols,
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

    @property
    def masked_frequencies(self):
        apply_mask_kernel = preprocessing_module.get_function("apply_mask")
        out_values = dispatch_kernel(
            apply_mask_kernel,
            self.n_rows,
            self.ncols,
            self.frequency_at_detector,
            self.undetectable_values_mask,
        )
        return out_values

    @property
    def masked_amplitudes(self):
        apply_mask_kernel = preprocessing_module.get_function("apply_mask")
        out_values = dispatch_kernel(
            apply_mask_kernel,
            self.n_rows,
            self.ncols,
            self.amplitude_at_detector,
            self.undetectable_values_mask,
        )
        return out_values

    def plot(self):
        bh_ax, boson_ax = cupy.meshgrid(self.BH_mass, self.boson_mass)
