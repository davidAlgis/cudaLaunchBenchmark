/**
 * @file kernels.cuh
 * @brief Declarations for toy CUDA kernels A, B, C, D used in launch benchmarks.
 *
 * These kernels perform arbitrary math so we can compare host-sequenced
 * launches vs dynamic parallelism without I/O or memory bottlenecks.
 *
 * Data flow:
 *   A -> B -> C -> D
 *
 * Build notes:
 * - Requires separable compilation (-rdc=true) if you use dynamic parallelism.
 * - All kernels accept an "iters" parameter to scale arithmetic work.
 *
 * company - Studio Nyx
 * Copyright (c) Studio Nyx. All rights reserved.
 */
#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace bench
{
/**
 * @brief Kernel A: seed compute. Writes out[i] from i and a lightweight mix.
 * @param out  Output buffer of size n.
 * @param n    Number of elements.
 * @param iters Amount of per-thread arithmetic work (>= 1).
 */
__global__ void kernelA(float* out, int n, int iters);

/**
 * @brief Kernel B: transforms A's output.
 * @param in   Input from A.
 * @param out  Output buffer of size n.
 * @param n    Number of elements.
 * @param iters Amount of per-thread arithmetic work (>= 1).
 */
__global__ void kernelB(const float* in, float* out, int n, int iters);

/**
 * @brief Kernel C: transforms B's output.
 * @param in   Input from B.
 * @param out  Output buffer of size n.
 * @param n    Number of elements.
 * @param iters Amount of per-thread arithmetic work (>= 1).
 */
__global__ void kernelC(const float* in, float* out, int n, int iters);

/**
 * @brief Kernel D: transforms C's output.
 * @param in   Input from C.
 * @param out  Output buffer of size n.
 * @param n    Number of elements.
 * @param iters Amount of per-thread arithmetic work (>= 1).
 */
__global__ void kernelD(const float* in, float* out, int n, int iters);
} // namespace bench
