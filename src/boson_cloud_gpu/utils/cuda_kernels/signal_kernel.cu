
/*
    CUDA ONLY SUPPORTS SINGLE PRECISION MATH!!
*/

extern "C" {

#define G 6.6743e-11
#define C 299792458.0
#define HBAR 1.0545718176461565e-34
#define m_sun 1.988409870698051e+30
#define om0 7.27220521664304e-05
#define R0 5.5e6
#define ONEV 1.60217653e-19
#define F_INT 1e30
#define DUTY 0.7
#define T_OBS 365 * 86400 * DUTY
#define NAN 0.0 / 0.0

#define ALPHA_MAX 0.1
#define FREQ_MAX 2000
#define FREQ_MIN 20

__global__ void get_signals(const float *bh_masses, const float *bh_ages_yrs,
                            const float *bh_spins, const float *distances,
                            const float *boson_masses, const int nrows,
                            const int ncols, float *out_frequencies,
                            float *out_amplitudes) {

  int x_abs = threadIdx.x + blockDim.x * blockIdx.x;
  int y_abs = threadIdx.y + blockDim.y * blockIdx.y;

  if ((x_abs < ncols) && (y_abs < nrows)) {
    int index = x_abs + ncols * y_abs;

    float bh_age_sec = bh_ages_yrs[x_abs] * 365 * 86400;
    float bh_mass = bh_masses[x_abs];
    float boson_mass = boson_masses[y_abs];
    float bh_spin = bh_spins[x_abs];
    float distance = distances[x_abs];

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

    float f_dot = 7e-15 + powf(1e-10 * (1e17 / F_INT), 4) *
                              powf(alpha / 0.1, 17) * (boson_mass / 1.0e-12) *
                              (boson_mass / 1.0e-12);

    float tau_inst =
        27 * 86400 / 10 * bh_mass / powf(alpha / 0.1, 9) / bh_spins[x_abs];

    if (tau_inst * 10 > bh_age_sec) {
      out_frequencies[index] = NAN;
      out_amplitudes[index] = NAN;
      return;
    }

    float tau_gw =
        27 * 86400 / 10 * bh_mass / powf(alpha / 0.1, 15) / bh_spins[x_abs];

    if ((tau_gw < 10 * bh_age_sec) || (10 * tau_inst > tau_gw)) {
      out_frequencies[index] = NAN;
      out_amplitudes[index] = NAN;
      return;
    }

    float frequency_at_source =
        483 * (boson_mass / 1.0e-12) *
        (1 - 0.0056 / 8 * powf((bh_mass / 10) * (boson_mass / 1.e-12), 2));

    float frequency_at_detector =
        frequency_at_source + f_dot * (bh_age_sec - tau_inst);

    if ((frequency_at_detector > FREQ_MAX) ||
        (frequency_at_detector < FREQ_MIN)) {
      out_frequencies[index] = NAN;
      out_amplitudes[index] = NAN;
      return;
    }

    float ceiled = ceil(frequency_at_detector / 10.) * 10.;
    float dfr = om0 * sqrtf(2 * ceiled * R0 / C);
    float df_dot = dfr / (2 * T_OBS / DUTY);

    if (df_dot < f_dot) {
      out_frequencies[index] = NAN;
      out_amplitudes[index] = NAN;
      return;
    }

    float amplitude_at_source =
        3.0e-24 / 10 * bh_mass * powf(alpha / 0.1, 7) * (bh_spin - chi_c) / 0.5;
    float timefactor = (1 + (bh_age_sec - tau_inst) / tau_gw);
    float amplitude_at_detector = amplitude_at_source / (timefactor * distance);

    // If arrived here all is masked
    out_frequencies[index] = frequency_at_detector;
    out_amplitudes[index] = amplitude_at_detector;
  }
}
}