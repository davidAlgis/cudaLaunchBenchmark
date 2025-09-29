cmake_minimum_required(VERSION 3.24)

# Project
project(cudaLaunchBench LANGUAGES CXX CUDA)

# Options
option(BENCH_ENABLE_WARNINGS "Enable extra warnings" ON)

# CUDA setup
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Default architectures if not provided by user.
# Override with: -DCMAKE_CUDA_ARCHITECTURES=86 (for RTX 30xx), 89 (RTX 40xx), etc.
if(NOT CMAKE_CUDA_ARCHITECTURES)
  # 70 = Volta baseline; tweak to your GPU for optimal codegen
  set(CMAKE_CUDA_ARCHITECTURES 70)
endif()

# Dynamic parallelism needs device linking and rdc
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
# (Implicitly sets -rdc=true for CUDA sources in this target)

# Warnings
if(BENCH_ENABLE_WARNINGS)
  if(MSVC)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/W4>)
  else()
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>)
  endif()
  # nvcc host-side warnings
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-Wall,-Wextra,-Wpedantic")
endif()

# Sources
set(BENCH_SRC
  cudaLaunchBench/main.cu
  cudaLaunchBench/kernels.cu
  cudaLaunchBench/launch_host.cu
  cudaLaunchBench/launch_dp.cu
)

add_executable(cudaLaunchBench ${BENCH_SRC})

target_include_directories(cudaLaunchBench PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/cudaLaunchBench
)

# Helpful defines you can use in code for conditional sections
target_compile_definitions(cudaLaunchBench PRIVATE
  BENCH_HAS_DYNAMIC_PARALLELISM=1
)

# On Windows, ensure static runtime only if you explicitly want it:
# set_property(TARGET cudaLaunchBench PROPERTY
#   MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>"
# )

# Print a friendly note if architectures may not support DP
# (dynamic parallelism requires sm_35+)
string(REPLACE ";" " " _archs "${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "CMAKE_CUDA_ARCHITECTURES = ${_archs}")
