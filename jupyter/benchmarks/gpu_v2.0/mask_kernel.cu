
extern "C" __global__ void
mask_array(const float *frequency, const float *amplitude, const float *tau_gw,
           const float *tau_inst, const float *bh_age_sec, const float *alpha,
           const float *spin, const float *chi_c, const float *f_dot,
           const float *df_dot, const int nrows, const int ncols,
           float *masked_frequency, float *masked_amplitude) {

  int x_abs = threadIdx.x + blockDim.x * blockIdx.x;
  int y_abs = threadIdx.y + blockDim.y * blockIdx.y;

  if ((x_abs < ncols) && (y_abs < nrows)) {

    if ((tau_gw[x_abs + ncols * y_abs] < 10. * bh_age_sec[x_abs]) ||
        (tau_inst[x_abs + ncols * y_abs] > bh_age_sec[x_abs] / 10) ||
        (alpha[x_abs + ncols * y_abs] > 0.1) ||
        (frequency[x_abs + ncols * y_abs] < 20.) ||
        (frequency[x_abs + ncols * y_abs] > 2048.) ||
        (10. * tau_inst[x_abs + ncols * y_abs] >
         tau_gw[x_abs + ncols * y_abs]) ||
        (spin[x_abs] < chi_c[x_abs + ncols * y_abs]) ||
        (df_dot[x_abs + ncols * y_abs] < f_dot[x_abs + ncols * y_abs])) {
      masked_frequency[x_abs + ncols * y_abs] = 0.0 / 0.0;
      masked_amplitude[x_abs + ncols * y_abs] = 0.0 / 0.0;
    } else {
      masked_frequency[x_abs + ncols * y_abs] =
          frequency[x_abs + ncols * y_abs];
      masked_amplitude[x_abs + ncols * y_abs] =
          amplitude[x_abs + ncols * y_abs];
    }
  }
}