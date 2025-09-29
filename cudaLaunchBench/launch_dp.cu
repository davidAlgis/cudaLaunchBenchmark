/**
 * @file launch_dp.cu
 * @brief Dynamic parallelism sequence for kernels A -> B -> C -> D.
 *
 * Strategy 2: a small "parent" kernel launches the four child kernels
 * from device side, queued on the tail of the same stream so they run
 * strictly in order without device-side blocking.
 *
 * Data flow:
 *   A(out=buf0)
 *   B(in=buf0, out=buf1)
 *   C(in=buf1, out=buf0)
 *   D(in=buf0, out=buf1)  // final result in buf1
 *
 * Build requirements:
 * - Compute capability sm_35+ (dynamic parallelism).
 * - Separable compilation / device linking enabled (-rdc=true).
 *
 * company - Studio Nyx
 * Copyright (c) Studio Nyx. All rights reserved.
 */
#include "kernels.cuh"

#include <cuda_runtime.h>
#include <stdexcept>

namespace bench
{

/**
 * @brief Parent kernel that launches A->B->C->D from device side.
 *
 * The launches use cudaStreamTailLaunch so they are appended to the same
 * execution stream in strict order. No device-side synchronization is
 * required; nested execution guarantees children complete before parent
 * returns.
 *
 * @param buf0     Device buffer used as output of A, input of B, output of C.
 * @param buf1     Device buffer used as output of B and final output of D.
 * @param n        Number of elements.
 * @param iters    Per-thread arithmetic work scaling (>= 1).
 * @param grid     Grid size to use for all child kernels.
 * @param block    Block size to use for all child kernels.
 */
__global__ void dp_sequence_kernel(float* buf0,
                                   float* buf1,
                                   int n,
                                   int iters,
                                   dim3 grid,
                                   dim3 block)
{
    // Use a single launcher thread to avoid duplicate child launches.
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        // A
        kernelA<<<grid, block, 0, cudaStreamTailLaunch>>>(buf0, n, iters);

        // B
        kernelB<<<grid, block, 0, cudaStreamTailLaunch>>>(buf0, buf1, n, iters);

        // C
        kernelC<<<grid, block, 0, cudaStreamTailLaunch>>>(buf1, buf0, n, iters);

        // D
        kernelD<<<grid, block, 0, cudaStreamTailLaunch>>>(buf0, buf1, n, iters);
    }
}

/**
 * @brief Host wrapper that launches the device-side sequence kernel.
 *
 * @param buf0    Device buffer used as output of A, input of B, output of C.
 * @param buf1    Device buffer used as output of B and final output of D.
 * @param n       Number of elements.
 * @param iters   Per-thread arithmetic work scaling (>= 1).
 * @param grid    Grid size for child kernels.
 * @param block   Block size for child kernels.
 * @param stream  CUDA stream for the parent kernel launch (default 0).
 *
 * Note: Synchronize the stream or use CUDA events around this call to time
 * dynamic parallelism end-to-end. The parent will not complete until all
 * children have finished (nested execution).
 */
void launch_dp_sequence(float* buf0,
                        float* buf1,
                        int n,
                        int iters,
                        dim3 grid,
                        dim3 block,
                        cudaStream_t stream /*=0*/)
{
    // Launch a single-thread parent that enqueues child kernels.
    dp_sequence_kernel<<<1, 1, 0, stream>>>(buf0, buf1, n, iters, grid, block);
    (void)cudaPeekAtLastError();
}

/**
 * @brief Convenience wrapper that auto-computes grid size from n and block size.
 *
 * @param buf0       Device buffer used as output of A, input of B, output of C.
 * @param buf1       Device buffer used as output of B and final output of D.
 * @param n          Number of elements.
 * @param iters      Per-thread arithmetic work scaling (>= 1).
 * @param blockSize  Threads per block for child kernels (default 256).
 * @param stream     CUDA stream for the parent kernel launch (default 0).
 */
void launch_dp_sequence_auto(float* buf0,
                             float* buf1,
                             int n,
                             int iters,
                             int blockSize /*=256*/,
                             cudaStream_t stream /*=0*/)
{
    if (blockSize <= 0)
    {
        blockSize = 256;
    }
    dim3 block(static_cast<unsigned>(blockSize), 1, 1);
    dim3 grid(static_cast<unsigned>((n + blockSize - 1) / blockSize), 1, 1);
    launch_dp_sequence(buf0, buf1, n, iters, grid, block, stream);
}

} // namespace bench
