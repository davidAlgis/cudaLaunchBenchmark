/**
 * @file main.cu
 * @brief Entry point for CUDA launch benchmark: host-sequenced vs dynamic parallelism.
 *
 * This program allocates two device buffers and runs a simple pipeline of four
 * kernels A -> B -> C -> D in two ways:
 *   1) Host-side sequential launches.
 *   2) Device-side dynamic parallelism (child kernels launched from a parent).
 *
 * It measures elapsed time with CUDA events, over multiple iterations, and
 * prints per-run timings and averages.
 *
 * Build requirements:
 * - CMake with CUDA separable compilation enabled (-rdc=true) for DP.
 * - Compute capability sm_35+ for dynamic parallelism.
 *
 * Usage examples:
 *   ./cudaLaunchBench
 *   ./cudaLaunchBench --n 1000000 --iters 128 --runs 20 --block 256
 *
 * company - Studio Nyx
 * Copyright (c) Studio Nyx. All rights reserved.
 */

#include "kernels.cuh"  // kernel signatures

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>
#include <string>
#include <iomanip>

// Prototypes from launchers (implemented in launch_host.cu and launch_dp.cu)
namespace bench
{
void launch_host_sequence_auto(float* buf0,
                               float* buf1,
                               int n,
                               int iters,
                               int blockSize = 256,
                               cudaStream_t stream = 0);

void launch_dp_sequence_auto(float* buf0,
                             float* buf1,
                             int n,
                             int iters,
                             int blockSize = 256,
                             cudaStream_t stream = 0);
} // namespace bench

/*--------------------------------------------------------------------*/
/*  Simple CLI parsing                                                */
/*--------------------------------------------------------------------*/
struct Args
{
    int n = 1 << 20;          // number of elements
    int iters = 64;           // arithmetic work per thread
    int runs = 10;            // measured repetitions per strategy
    int warmup = 3;           // warmup runs per strategy
    int block = 256;          // threads per block
    int device = 0;           // CUDA device id
};

static bool parse_int_flag(const char* flag, int& dst, int argc, char** argv)
{
    for (int i = 1; i < argc; ++i)
    {
        if (std::strcmp(argv[i], flag) == 0 && i + 1 < argc)
        {
            dst = std::atoi(argv[i + 1]);
            return true;
        }
    }
    return false;
}

static Args parse_args(int argc, char** argv)
{
    Args a;
    (void)parse_int_flag("--n", a.n, argc, argv);
    (void)parse_int_flag("--iters", a.iters, argc, argv);
    (void)parse_int_flag("--runs", a.runs, argc, argv);
    (void)parse_int_flag("--warmup", a.warmup, argc, argv);
    (void)parse_int_flag("--block", a.block, argc, argv);
    (void)parse_int_flag("--device", a.device, argc, argv);
    return a;
}

/*--------------------------------------------------------------------*/
/*  CUDA helpers                                                      */
/*--------------------------------------------------------------------*/
static void check_cuda(cudaError_t e, const char* what)
{
    if (e != cudaSuccess)
    {
        std::cerr << "CUDA error at " << what << ": "
                  << cudaGetErrorString(e) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

static float time_with_events(std::function<void(cudaStream_t)> work,
                              cudaStream_t stream)
{
    cudaEvent_t beg, end;
    check_cuda(cudaEventCreate(&beg), "cudaEventCreate(beg)");
    check_cuda(cudaEventCreate(&end), "cudaEventCreate(end)");
    check_cuda(cudaEventRecord(beg, stream), "cudaEventRecord(beg)");
    work(stream);
    check_cuda(cudaEventRecord(end, stream), "cudaEventRecord(end)");
    check_cuda(cudaEventSynchronize(end), "cudaEventSynchronize(end)");
    float ms = 0.0f;
    check_cuda(cudaEventElapsedTime(&ms, beg, end), "cudaEventElapsedTime");
    cudaEventDestroy(beg);
    cudaEventDestroy(end);
    return ms;
}

/*--------------------------------------------------------------------*/
/*  Main                                                              */
/*--------------------------------------------------------------------*/
int main(int argc, char** argv)
{
    Args args = parse_args(argc, argv);

    // Select device and report capabilities.
    int deviceCount = 0;
    check_cuda(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount");
    if (deviceCount == 0)
    {
        std::cerr << "No CUDA devices found." << std::endl;
        return EXIT_FAILURE;
    }
    if (args.device < 0 || args.device >= deviceCount)
    {
        std::cerr << "Invalid --device " << args.device
                  << " (have " << deviceCount << " devices)" << std::endl;
        return EXIT_FAILURE;
    }
    check_cuda(cudaSetDevice(args.device), "cudaSetDevice");

    cudaDeviceProp props{};
    check_cuda(cudaGetDeviceProperties(&props, args.device), "cudaGetDeviceProperties");
    const int sm = props.major * 10 + props.minor;
    const bool dp_supported = (props.major > 3) || (props.major == 3 && props.minor >= 5);

    std::cout << "Device: " << props.name
              << " (SM " << props.major << "." << props.minor << ", "
              << props.multiProcessorCount << " SMs)"
              << std::endl;
    std::cout << "Settings: n=" << args.n
              << " iters=" << args.iters
              << " runs=" << args.runs
              << " warmup=" << args.warmup
              << " block=" << args.block
              << " device=" << args.device
              << std::endl;

    // Allocate buffers.
    float* d_buf0 = nullptr;
    float* d_buf1 = nullptr;
    size_t bytes = static_cast<size_t>(args.n) * sizeof(float);
    check_cuda(cudaMalloc(&d_buf0, bytes), "cudaMalloc(buf0)");
    check_cuda(cudaMalloc(&d_buf1, bytes), "cudaMalloc(buf1)");

    // Create a stream for all work.
    cudaStream_t stream = nullptr;
    check_cuda(cudaStreamCreate(&stream), "cudaStreamCreate");

    // Warmup host sequence.
    for (int i = 0; i < args.warmup; ++i)
    {
        bench::launch_host_sequence_auto(d_buf0, d_buf1, args.n, args.iters, args.block, stream);
    }
    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize (host warmup)");

    // Measure host sequence.
    std::vector<float> host_ms;
    host_ms.reserve(args.runs);
    for (int i = 0; i < args.runs; ++i)
    {
        float ms = time_with_events(
            [&](cudaStream_t s)
            {
                bench::launch_host_sequence_auto(d_buf0, d_buf1, args.n, args.iters, args.block, s);
            },
            stream);
        host_ms.push_back(ms);
    }

    // Warmup dynamic parallelism if supported.
    std::vector<float> dp_ms;
    bool dp_ran = false;
    if (dp_supported)
    {
        for (int i = 0; i < args.warmup; ++i)
        {
            bench::launch_dp_sequence_auto(d_buf0, d_buf1, args.n, args.iters, args.block, stream);
        }
        check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize (dp warmup)");

        // Measure DP.
        dp_ran = true;
        dp_ms.reserve(args.runs);
        for (int i = 0; i < args.runs; ++i)
        {
            float ms = time_with_events(
                [&](cudaStream_t s)
                {
                    bench::launch_dp_sequence_auto(d_buf0, d_buf1, args.n, args.iters, args.block, s);
                },
                stream);
            dp_ms.push_back(ms);
        }
    }
    else
    {
        std::cout << "Dynamic parallelism not supported on this device (requires SM 3.5+)."
                  << std::endl;
    }

    // Reduce results.
    auto mean = [](const std::vector<float>& v) -> double
    {
        if (v.empty())
        {
            return 0.0;
        }
        double sum = 0.0;
        for (float x : v)
        {
            sum += static_cast<double>(x);
        }
        return sum / static_cast<double>(v.size());
    };

    auto minv = [](const std::vector<float>& v) -> double
    {
        if (v.empty())
        {
            return 0.0;
        }
        double m = v.front();
        for (float x : v)
        {
            if (x < m)
            {
                m = x;
            }
        }
        return m;
    };

    auto maxv = [](const std::vector<float>& v) -> double
    {
        if (v.empty())
        {
            return 0.0;
        }
        double m = v.front();
        for (float x : v)
        {
            if (x > m)
            {
                m = x;
            }
        }
        return m;
    };

    // Print per-run and summary.
    std::cout << std::fixed << std::setprecision(3);

    std::cout << "\nHost-sequenced runs (ms):";
    for (size_t i = 0; i < host_ms.size(); ++i)
    {
        std::cout << (i == 0 ? " " : ", ") << host_ms[i];
    }
    std::cout << "\nHost-sequenced avg/min/max (ms): "
              << mean(host_ms) << " / " << minv(host_ms) << " / " << maxv(host_ms)
              << std::endl;

    if (dp_ran)
    {
        std::cout << "\nDynamic-parallelism runs (ms):";
        for (size_t i = 0; i < dp_ms.size(); ++i)
        {
            std::cout << (i == 0 ? " " : ", ") << dp_ms[i];
        }
        std::cout << "\nDynamic-parallelism avg/min/max (ms): "
                  << mean(dp_ms) << " / " << minv(dp_ms) << " / " << maxv(dp_ms)
                  << std::endl;

        // Relative comparison.
        double avg_host = mean(host_ms);
        double avg_dp = mean(dp_ms);
        if (avg_dp > 0.0)
        {
            double ratio = avg_host / avg_dp;
            std::cout << "\nSpeed ratio (host / dp): " << ratio << "x" << std::endl;
        }
    }

    // Cleanup.
    cudaStreamDestroy(stream);
    cudaFree(d_buf0);
    cudaFree(d_buf1);

    return 0;
}
