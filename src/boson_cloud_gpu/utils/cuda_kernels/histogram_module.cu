
extern "C" {
// Credits to  Nikolay Sakharnykh
// https://developer.nvidia.com/blog/gpu-pro-tip-fast-histograms-using-shared-atomics-maxwell/

__global__ void
histogram_per_block_atomics(const float *value, const float *weights,
                            const int *n_bins, const float *step_sizes,
                            const float *start_values, const int nrows,
                            const int ncols, int *counts_per_block) {
  // pixel coordinates
  int x_abs = blockIdx.x * blockDim.x + threadIdx.x;
  int y_abs = blockIdx.y * blockDim.y + threadIdx.y;

  if ((x_abs < ncols) && (y_abs < nrows)) {

    // unique linear index of the thread, within a 2D block
    int idx = threadIdx.x + threadIdx.y * blockDim.x;

    int threads_per_block = blockDim.x * blockDim.y;

    // initialize temporary accumulation array in shared memory
    /*
        Since the number of bins can be bigger than the number of threads in a
        single block, a loop is made to be sure that every element of the array
    is initialize one, and only one time.
    */
    __shared__ int block_temp_histogram[n_bins[y_abs] * nrows];
    int i;
    for (int i = idx; i < n_bins[y_abs]; i += threads_per_block) {
      block_temp_histogram[i] = 0;
    }
    __syncthreads();

    // process pixels
    int bin_index = (int)((value[x_abs + y_abs * ncols] - start_values[y_abs]) /
                          step_sizes[y_abs]);
    atomicAdd(&block_temp_histogram[bin_index], 1);

    __syncthreads();

    // return the partial histograms to global memory
    for (i = idx; i < n_bins[y_abs]; i += threads_per_block) {
      counts_per_block[i + y_abs * ncols] = block_temp_histogram[i];
    }
  }
}

__global__ void histogram_global(const int *partial_histograms,
                                 const int *n_bins, const int nrows,
                                 int *histogram, int *out_histogram) {
  int y_abs = blockIdx.y * blockDim.y + threadIdx.y;

  if (y_abs < nrows) {
    int count = 0;
    int i;
    for (i = 0; i <)
      i histogram[y_abs]
  }
}
}