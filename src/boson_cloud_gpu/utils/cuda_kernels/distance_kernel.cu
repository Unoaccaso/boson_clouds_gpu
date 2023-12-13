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

extern "C" __global__ void distance(const float *position,
                                    const int n_positions, float *distance) {
  int x_abs = threadIdx.x + blockDim.x * blockIdx.x;

  if (x_abs < n_positions) {
    distance[x_abs] = sqrt(position[x_abs * 3] * position[x_abs * 3] +
                           position[x_abs * 3 + 1] * position[x_abs * 3 + 1] +
                           position[x_abs * 3 + 2] * position[x_abs * 3 + 2]);
  }
}