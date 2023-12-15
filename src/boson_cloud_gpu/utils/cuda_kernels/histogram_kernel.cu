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
#define NAN 0. / 0.
__global__ void make_bins(const float *mean_frequency, const int band_size,
                          const float t_fft, const int nrows, const int nbins,
                          float *frequency_bins) {

  int x = threadIdx.x + blockDim.x * blockIdx.x;
  int y = threadIdx.y + blockDim.y * blockIdx.y;

  if ((x < nbins) && (y < nrows)) {
    int index = x + nbins * y;

    int checknan = isnan(mean_frequency[y]);
    if (checknan) {
      frequency_bins[index] = NAN;
    } else {
      // F0 is the mean frequency of the distribution, it is used as a center
      // point for the histogram. The actual center point will be the closest
      // power of 10, for better compatibility with older algorithms.

      float start_frequency = mean_frequency[y] - band_size / 2;

      frequency_bins[index] = start_frequency + float(x / t_fft);
    }
  }
}

__global__ void make_histograms(const float *frequencies, const int *amplitudes,
                                const float *mean_frequency,
                                const int band_size, const float t_fft,
                                const int nbins, const int ncols,
                                const int nrows, float *counts) {

  int x = threadIdx.x + blockDim.x * blockIdx.x;
  int y = threadIdx.y + blockDim.y * blockIdx.y;
  if ((x < ncols) && (y < nrows)) {
    int index = x + y * ncols;

    // The first check excludes all the empty rows
    int checknan_mean = isnan(mean_frequency[y]);
    // Second check skips the matrix element if its value is nan
    int checknan_frequency = isnan(frequencies[index]);
    if (checknan_mean || checknan_frequency) {
      return;
    } else {

      // F0 is the mean frequency of the distribution, it is used as a center
      // point for the histogram. The actual center point will be the closest
      // power of 10, for better compatibility with older algorithms.
      float start_frequency = mean_frequency[y] - band_size / 2;

      int bin_id = (int)((frequencies[index] - start_frequency) * t_fft);

      atomicAdd(&counts[bin_id + y * nbins], 1);
    }
  }
  __syncthreads();
}
}