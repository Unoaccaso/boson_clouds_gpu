extern "C" __global__ void distance(const float *position,
                                    const int n_positions, float *distance) {
  int y_abs = threadIdx.y + blockDim.y * blockIdx.y;

  if (y_abs < n_positions) {
    distance[y_abs] = sqrt(position[y_abs * 3] * position[y_abs * 3] +
                           position[y_abs * 3 + 1] * position[y_abs * 3 + 1] +
                           position[y_abs * 3 + 2] * position[y_abs * 3 + 2]);
  }
}