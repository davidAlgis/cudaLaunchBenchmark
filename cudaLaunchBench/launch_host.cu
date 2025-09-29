/**
 * @file launch_host.cu
 * @brief Host-side sequential launches for kernels A -> B -> C -> D.
 *
 * This file implements the "strategy 1" baseline: the host launches
 * four kernels one after another on the same CUDA stream.
 *
 * Usage pattern (example):
 *   // Device buffers (ping-pong):
 *   float* d_buf0; // size n
 *   float* d_buf1; // size n
 *   // ... allocate ...
 *   bench::launch_host_sequence_auto(d_buf0, d_buf1, n, iters);
 *   // After the call, results are in d_buf1.
 *
 * Notes:
 * - No device-wide synchronization is performed here; the caller should
 *   place CUDA events or synchronize the stream as needed for timing.
 * - Errors are checked with cudaPeekAtLastError() after each launch.
 * - Grid/block can be user-provided or auto-computed.
 *
 * company - Studio Nyx
 * Copyright (c) Studio Nyx. All rights reserved.
 */
#include "kernels.cuh"

#include <cuda_runtime.h>
#include <stdexcept>

namespace bench {

/**
 * @brief Launch A->B->C->D sequentially on the given stream.
 *
 * @param buf0    Device buffer used as output of A, input of B, output of C.
 * @param buf1    Device buffer used as output of B and final output of D.
 * @param n       Number of elements.
 * @param iters   Per-thread arithmetic work scaling (>= 1).
 * @param grid    Grid size for all kernels.
 * @param block   Block size for all kernels.
 * @param stream  CUDA stream (default 0).
 *
 * The data flow is:
 *   A(out=buf0)
 *   B(in=buf0, out=buf1)
 *   C(in=buf1, out=buf0)
 *   D(in=buf0, out=buf1)  // final result in buf1
 */
void launch_host_sequence(float *buf0, float *buf1, int n, int iters, dim3 grid,
                          dim3 block, cudaStream_t stream /*= 0*/) {
  // Kernel A
  kernelA<<<grid, block, 0, stream>>>(buf0, n, iters);
  (void)cudaPeekAtLastError();

  // Kernel B
  kernelB<<<grid, block, 0, stream>>>(buf0, buf1, n, iters);
  (void)cudaPeekAtLastError();

  // Kernel C
  kernelC<<<grid, block, 0, stream>>>(buf1, buf0, n, iters);
  (void)cudaPeekAtLastError();

  // Kernel D
  kernelD<<<grid, block, 0, stream>>>(buf0, buf1, n, iters);
  (void)cudaPeekAtLastError();
}

/**
 * @brief Convenience wrapper that auto-computes grid size from n and block
 * size.
 *
 * @param buf0      Device buffer used as output of A, input of B, output of C.
 * @param buf1      Device buffer used as output of B and final output of D.
 * @param n         Number of elements.
 * @param iters     Per-thread arithmetic work scaling (>= 1).
 * @param blockSize Threads per block (default 256).
 * @param stream    CUDA stream (default 0).
 */
void launch_host_sequence_auto(float *buf0, float *buf1, int n, int iters,
                               int blockSize /*=256*/,
                               cudaStream_t stream /*=0*/) {
  if (blockSize <= 0) {
    blockSize = 256;
  }
  dim3 block(static_cast<unsigned>(blockSize), 1, 1);
  dim3 grid(static_cast<unsigned>((n + blockSize - 1) / blockSize), 1, 1);
  launch_host_sequence(buf0, buf1, n, iters, grid, block, stream);
}

} // namespace bench
