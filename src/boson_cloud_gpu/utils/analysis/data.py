# ADDLICENSE

import cupy

from functools import cached_property


class Histogram:
    def __init__(
        self,
        frequencies,
        amplitudes,
        t_fft,
    ) -> None:
        self._frequencies = frequencies
        self._amplitudes = amplitudes
        self._t_fft = t_fft

    @property
    def frequencies(self):
        return self._frequencies

    @property
    def amplitudes(self):
        return self._amplitudes

    @property
    def t_fft(self):
        return self._t_fft

    @cached_property
    def histogram_kernel(self):
        kernel = cupy.RawKernel(
            r"""
            extern "C" __global__
            void histogram(float* freqs, float* amps, float* bins){
                
            }
            """,
            "histogram",
        )
        return kernel

    def histogram(self):
        n_rows = self.frequencies.empty_shape
