/**
 * @file kernels.cu
 * @brief Definitions for toy CUDA kernels A, B, C, D used in launch benchmarks.
 *
 * The math is intentionally arbitrary but deterministic. The "iters" parameter
 * scales ALU work so you can observe scheduling/launch overhead differences
 * between host-side sequencing and dynamic parallelism.
 *
 * company - Studio Nyx
 * Copyright (c) Studio Nyx. All rights reserved.
 */
#include "kernels.cuh"

#include <cuda.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cmath>

namespace bench
{
/**
 * @brief Small deterministic hash to vary inputs across threads.
 */
__device__ __forceinline__ float hash_u32(uint32_t x)
{
    // Thomas Wang style mix, mapped to [0,1).
    x ^= 61u;
    x ^= x >> 16;
    x *= 9u;
    x ^= x >> 4;
    x *= 0x27d4eb2du;
    x ^= x >> 15;
    const float uf = static_cast<float>(x) * (1.0f / 4294967296.0f);
    return uf;
}

/**
 * @brief A bit of ALU work to keep cores busy without memory pressure.
 */
__device__ __forceinline__ float work_unit(float v, int iters)
{
    // Clamp to at least one iteration to avoid being optimized out.
    const int kIters = iters > 0 ? iters : 1;
    float a = v;
    float b = v * 0.5f + 0.1234567f;
    for (int k = 0; k < kIters; ++k)
    {
        // Mix in some transcendental ops and fused multiply-adds.
        float s = __sinf(a);
        float c = __cosf(b);
        a = __fmaf_rn(a, 1.000173f, 0.6180339f) + s * 0.75f - c * 0.25f;
        b = __fdividef(b + 0.0001f, __fsqrt_rn(fabsf(a) + 1.0f));
        // Keep values in a reasonable range.
        a = fmodf(a, 3.1415926f);
        b = fmodf(b, 3.1415926f);
    }
    return a + 0.5f * b;
}

/*======================================================================
  Kernel A
======================================================================*/
__global__ void kernelA(float* out, int n, int iters)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
    {
        return;
    }
    const float seed = hash_u32(static_cast<uint32_t>(i)) * 6.2831853f; // 2*pi
    const float v = work_unit(seed + static_cast<float>(i) * 0.001f, iters);
    out[i] = v;
}

/*======================================================================
  Kernel B
======================================================================*/
__global__ void kernelB(const float* in, float* out, int n, int iters)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
    {
        return;
    }
    const float x = in[i] + 0.1f;
    const float v = work_unit(x, iters) + __fmaf_rn(x, x, 0.0f) * 0.01f;
    out[i] = v;
}

/*======================================================================
  Kernel C
======================================================================*/
__global__ void kernelC(const float* in, float* out, int n, int iters)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
    {
        return;
    }
    const float x = in[i];
    const float v = work_unit(x * 0.75f - 0.2f, iters);
    // Blend a bit of nonlinearity.
    out[i] = v + x * x * 0.005f - __sinf(x) * 0.25f;
}

/*======================================================================
  Kernel D
======================================================================*/
__global__ void kernelD(const float* in, float* out, int n, int iters)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
    {
        return;
    }
    const float x = in[i];
    const float v = work_unit(x + 0.3141592f, iters);
    // Final mapping to keep values well-behaved.
    out[i] = __fsqrt_rn(fabsf(v)) + 0.1f * x;
}

} // namespace bench
