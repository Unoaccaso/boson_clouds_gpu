/*
Copyright(C) 2023
Riccardo Felicetti(felicettiriccardo1 @gmail.com)

This program is free software : you can redistribute it and / or modify
it under the terms of the GNU AFFERO GENERAL PUBLIC LICENSE Version 3,        \
    19 November 2007

Copyright(C) 2007 Free Software Foundation, Inc.< https: // fsf.org/>
Everyone is permitted to copy and distribute verbatim copies
of this license document, but changing it is not allowed.

You should have received a copy of theGNU AFFERO GENERAL PUBLIC LICENSE
along with this program.If not, see < http: // www.gnu.org/licenses/>.
*/

extern "C" {
#define NAN 0 / 0
__global__ void make_bins(const float *median_frequencies, const int band_size,
                          const float t_fft, const int nrows, const int nbins,
                          float *bins) {

  int x_abs = threadIdx.x + blockDim.x * blockIdx.x;
  int y_abs = threadIdx.y + blockDim.y * blockIdx.y;

  if ((x_abs < nbins) && (y_abs < nrows)) {
    int index = x_abs + nbins * y_abs;

    float start_frequency = median_frequencies[y_abs] - band_size / 2;
    int checknan = isnan(start_frequency);
    if (checknan) {
      bins[index] = NAN;
    } else {
      bins[index] = start_frequency + float(x_abs / t_fft);
    }
  }
}
}