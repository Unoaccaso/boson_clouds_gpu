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

#define G (float)(6.6743e-11)
#define PI (float)(3.14159265)
#define C (float)(299792458.0)
#define HBAR (float)(1.05457182e-34)
#define m_sun (float)(1.98840987e+30)
#define OM0 (float)(7.2722052e-05)
#define R0 (float)(5.5e6)
#define ONEV (float)(1.6021765e-19)
#define F_INT (float)(1e30)
#define DUTY (float)(0.7)
#define T_OBS (float)(365 * 86400 * DUTY)
#define NAN 0.0 / 0.0

#define POW powf

#define ALPHA_MAX 0.1
#define FREQ_MAX 2000
#define FREQ_MIN 20

#define TILE_DIM_X 32
#define TILE_DIM_Y 32

__global__ void get_signals(const float *bh_masses, const float *bh_ages_yrs,
                            const float *bh_spins, const float *distances,
                            const float *boson_masses, const int nrows,
                            const int ncols, float *out_frequencies,
                            float *out_amplitudes) {

  int x_abs = threadIdx.x + blockDim.x * blockIdx.x;
  int y_abs = threadIdx.y + blockDim.y * blockIdx.y;

  __shared__ float bh_masses_tile[TILE_DIM_X];
  __shared__ float boson_masses_tile[TILE_DIM_Y];
  __shared__ float bh_spins_tile[TILE_DIM_X];
  __shared__ float distances_tile[TILE_DIM_X];
  __shared__ float bh_ages_sec_tile[TILE_DIM_X];

  bh_ages_sec_tile[threadIdx.x] = bh_ages_yrs[x_abs] * 365 * 86400;
  bh_masses_tile[threadIdx.x] = bh_masses[x_abs];
  boson_masses_tile[threadIdx.y] = boson_masses[y_abs];
  bh_spins_tile[threadIdx.x] = bh_spins[x_abs];
  distances_tile[threadIdx.x] = distances[x_abs];
  __syncthreads();

  if ((x_abs < ncols) && (y_abs < nrows)) {
    // Skipping the first row as it is reserved for minimum frequencies value
    unsigned int index = x_abs + ncols * y_abs;

    float bh_mass = bh_masses_tile[threadIdx.x];
    float boson_mass = boson_masses_tile[threadIdx.y];
    float bh_spin = bh_spins_tile[threadIdx.x];
    float distance = distances_tile[threadIdx.x];
    float bh_age_sec = bh_ages_sec_tile[threadIdx.x];

    float alpha = G / (C * C * C * HBAR) * 2e30 * bh_mass * boson_mass * ONEV;
    if (alpha > ALPHA_MAX) {
      out_frequencies[index] = NAN;
      out_amplitudes[index] = NAN;
      return;
    }

    float chi_c = 4 * alpha / (1 + 4 * alpha * alpha);
    if (bh_spin < chi_c) {
      out_frequencies[index] = NAN;
      out_amplitudes[index] = NAN;
      return;
    }

    float f_dot = 7e-15 * POW(alpha / 0.1, 17) * POW(boson_mass / 1.0e-12, 2);
    float f_dot2 = 1e-10 * POW(1E17 / F_INT, 4) * POW(alpha / 0.1, 17) *
                   POW(boson_mass / 1.0e-12, 2);

    f_dot += f_dot2;

    float tau_inst = 27 * 86400 / 10 * bh_mass / POW(alpha / 0.1, 9) / bh_spin;

    if (tau_inst > bh_age_sec / 10) {
      out_frequencies[index] = NAN;
      out_amplitudes[index] = NAN;
      return;
    }

    float tau_gw =
        6.5E4 * 365 * 86400 * bh_mass / 10 / POW(alpha / 0.1, 15) / bh_spin;

    if ((tau_gw < 10 * bh_age_sec) || (tau_inst / 10 > tau_gw)) {
      out_frequencies[index] = NAN;
      out_amplitudes[index] = NAN;
      return;
    }

    float frequency_at_source =
        483 * (boson_mass / 1.0e-12) *
        (1 - 0.0056 / 8 * POW((bh_mass / 10) * (boson_mass / 1.e-12), 2));

    float freq_max = C * C * C / (2 * PI * G * 2e30 * bh_mass) * bh_spin /
                     (1 + sqrtf(1 - bh_spin * bh_spin));

    if (frequency_at_source > freq_max) {
      out_frequencies[index] = NAN;
      out_amplitudes[index] = NAN;
      return;
    }

    float frequency_at_detector =
        frequency_at_source + f_dot * (bh_age_sec - tau_inst);

    if ((frequency_at_detector > FREQ_MAX) ||
        (frequency_at_detector < FREQ_MIN)) {
      out_frequencies[index] = NAN;
      out_amplitudes[index] = NAN;
      return;
    }

    float ceiled = ceil(frequency_at_detector / 10) * 10;
    float dfr = OM0 * sqrtf(2 * ceiled * R0 / C);
    float df_dot = dfr / (2 * T_OBS / DUTY);

    if (df_dot < f_dot) {
      out_frequencies[index] = NAN;
      out_amplitudes[index] = NAN;
      return;
    }

    float amplitude_at_source =
        3.0e-24 / 10 * bh_mass * POW(alpha / 0.1, 7) * (bh_spin - chi_c) / 0.5;
    float timefactor = (1 + (bh_age_sec - tau_inst) / tau_gw);
    float amplitude_at_detector = amplitude_at_source / (timefactor * distance);

    // If arrived here all is masked
    out_frequencies[index] = frequency_at_detector;
    out_amplitudes[index] = amplitude_at_detector;
  }
}
}