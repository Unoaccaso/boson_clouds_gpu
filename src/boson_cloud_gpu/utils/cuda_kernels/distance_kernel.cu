extern "C" __global__ void distance(const float *position,
                                    const int n_positions, float *distance) {
  int x_abs = threadIdx.x + blockDim.x * blockIdx.x;

  if (x_abs < n_positions) {
    distance[x_abs] = sqrt(position[x_abs * 3] * position[x_abs * 3] +
                           position[x_abs * 3 + 1] * position[x_abs * 3 + 1] +
                           position[x_abs * 3 + 2] * position[x_abs * 3 + 2]);
  }
}