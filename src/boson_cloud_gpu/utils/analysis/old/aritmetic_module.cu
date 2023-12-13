
extern "C" {
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
#define NAN 0.0 / 0.0

#define ALPHA_MAX 0.1
#define FREQ_MAX 2000
#define FREQ_MIN 20

__global__ void alpha(const float *bh_mass, const float *boson_mass,
                      const int nrows, const int ncols, float *_alpha) {

  int x_abs = threadIdx.x + blockDim.x * blockIdx.x;
  int y_abs = threadIdx.y + blockDim.y * blockIdx.y;

  if ((x_abs < ncols) && (y_abs < nrows)) {
    int index = x_abs + ncols * y_abs;
    float alpha = G / (c * c * c * hbar) * 2e30 * bh_mass[x_abs] *
                  boson_mass[y_abs] * onev;
    if (alpha < ALPHA_MAX) {
      _alpha[index] = alpha;
    } else {
      _alpha[index] = NAN;
    }
  }
}

__global__ void f_dot(const float *alpha, const float *boson_mass,
                      const int nrows, const int ncols, float *f_dot) {

  int x_abs = threadIdx.x + blockDim.x * blockIdx.x;
  int y_abs = threadIdx.y + blockDim.y * blockIdx.y;

  if ((x_abs < ncols) && (y_abs < nrows)) {

    // faster power to 17 ~ half the time
    float alpha_17 = alpha[x_abs + ncols * y_abs] / 0.1;
    int k;
    for (k = 0; k < 4; k++) {
      alpha_17 *= alpha_17;
    }
    alpha_17 *= alpha[x_abs + ncols * y_abs] / 0.1;

    float A = 1e-10 * (1e17 / f_int);
    float A_4 = A * A * A * A;

    f_dot[x_abs + ncols * y_abs] = 7e-15 + A_4 * alpha_17 *
                                               (boson_mass[y_abs] / 1.0e-12) *
                                               (boson_mass[y_abs] / 1.0e-12);
  }
}

__global__ void df_dot(const float *frequency, const int nrows, const int ncols,
                       float *df_dot) {

  int x_abs = threadIdx.x + blockDim.x * blockIdx.x;
  int y_abs = threadIdx.y + blockDim.y * blockIdx.y;

  if ((x_abs < ncols) && (y_abs < nrows)) {
    float ceiled = ceil(frequency[x_abs + ncols * y_abs] / 10.) * 10.;
    float dfr = om0 * sqrt(2 * ceiled * r0 / c);
    df_dot[x_abs + ncols * y_abs] = dfr / (2 * t_obs / duty);
  }
}

__global__ void tau_inst(const float *bh_mass, const float *spin,
                         const float *alpha, const int nrows, const int ncols,
                         float *tau_inst) {

  int x_abs = threadIdx.x + blockDim.x * blockIdx.x;
  int y_abs = threadIdx.y + blockDim.y * blockIdx.y;

  if ((x_abs < ncols) && (y_abs < nrows)) {

    // faster power to 9 ~ half the time
    float alpha_9 = alpha[x_abs + ncols * y_abs] / 0.1;
    int k;
    for (k = 0; k < 3; k++) {
      alpha_9 *= alpha_9;
    }
    alpha_9 *= alpha[x_abs + ncols * y_abs] / 0.1;

    tau_inst[x_abs + ncols * y_abs] =
        27 * 86400 / 10 * bh_mass[x_abs] / alpha_9 / spin[x_abs];
  }
}

__global__ void tau_gw(const float *bh_mass, const float *spin,
                       const float *alpha, const int nrows, const int ncols,
                       float *tau_gw) {

  int x_abs = threadIdx.x + blockDim.x * blockIdx.x;
  int y_abs = threadIdx.y + blockDim.y * blockIdx.y;

  if ((x_abs < ncols) && (y_abs < nrows)) {

    // faster power to 15 ~ half the time
    float alpha_15 = alpha[x_abs + ncols * y_abs] / 0.1;
    int k;
    for (k = 0; k < 4; k++) {
      alpha_15 *= alpha_15;
    }
    alpha_15 /= alpha[x_abs + ncols * y_abs] / 0.1;

    tau_gw[x_abs + ncols * y_abs] =
        6.5e4 * 365 * 86400 * bh_mass[x_abs] / 10 / alpha_15 / spin[x_abs];
  }
}

__global__ void chi_c(const float *alpha, const int nrows, const int ncols,
                      float *chi_c) {

  int x_abs = threadIdx.x + blockDim.x * blockIdx.x;
  int y_abs = threadIdx.y + blockDim.y * blockIdx.y;

  if ((x_abs < ncols) && (y_abs < nrows)) {
    float _alpha = alpha[x_abs + ncols * y_abs];
    chi_c[x_abs + ncols * y_abs] = 4 * _alpha / (1 + 4 * _alpha * _alpha);
  }
}

__global__ void frequency_at_detector(const float *bh_mass,
                                      const float *boson_mass,
                                      const float *f_dot, const float *tau_inst,
                                      const float *bh_age_sec, const int nrows,
                                      const int ncols, float *frequency) {

  int x_abs = threadIdx.x + blockDim.x * blockIdx.x;
  int y_abs = threadIdx.y + blockDim.y * blockIdx.y;

  if ((x_abs < ncols) && (y_abs < nrows)) {
    int index = x_abs + ncols * y_abs;
    float emitted_freq =
        483 * (boson_mass[y_abs] / 1.0e-12) *
        (1 - 0.0056 / 8 * (bh_mass[x_abs] / 10) * (boson_mass[y_abs] / 1.e-12) *
                 (bh_mass[x_abs] / 10) * (boson_mass[y_abs] / 1.e-12));

    if ((emitted_freq > FREQ_MIN) && (emitted_freq < FREQ_MAX) &&
        (bh_age_sec[x_abs] > 10 * tau_inst[index])) {
      frequency[index] =
          emitted_freq + f_dot[index] * (bh_age_sec[x_abs] - tau_inst[index]);
    } else {
      frequency[index] = NAN;
    }
  }
}

__global__ void amplitude_at_detector(
    const float *bh_mass, const float *boson_mass, const float *spin,
    const float *bh_age_sec, const float *distance, const float *alpha,
    const float *tau_inst, const float *tau_gw, const float *chi_c,
    const int nrows, const int ncols, float *amplitude) {

  int x_abs = threadIdx.x + blockDim.x * blockIdx.x;
  int y_abs = threadIdx.y + blockDim.y * blockIdx.y;

  if ((x_abs < ncols) && (y_abs < nrows)) {

    // faster power to 7 ~ half the time
    float alpha_7 = alpha[x_abs + ncols * y_abs] / 0.1;
    int k;
    for (k = 0; k < 3; k++) {
      alpha_7 *= alpha_7;
    }
    alpha_7 /= alpha[x_abs + ncols * y_abs] / 0.1;

    float timefactor =
        (1 + (bh_age_sec[x_abs] - tau_inst[x_abs + ncols * y_abs]) /
                 tau_gw[x_abs + ncols * y_abs]);

    float amp_at_1kpc = 3.0e-24 / 10 * bh_mass[x_abs] * alpha_7 *
                        (spin[x_abs] - chi_c[x_abs + ncols * y_abs]) / 0.5;

    amplitude[x_abs + ncols * y_abs] =
        amp_at_1kpc / (timefactor * distance[x_abs]);
  }
}
}