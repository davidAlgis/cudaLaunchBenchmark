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

# Architectures (override with -DCMAKE_CUDA_ARCHITECTURES=86, 89, etc.)
if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

# Dynamic parallelism needs separable compilation (-rdc=true)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

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
target_compile_definitions(cudaLaunchBench PRIVATE
  BENCH_HAS_DYNAMIC_PARALLELISM=1
)

# ------------------------------
# Warnings per toolchain (FIX)
# ------------------------------
if(BENCH_ENABLE_WARNINGS)
  if(MSVC)
    # C++ files: /W4 ; CUDA files: pass /W4 to host compiler via -Xcompiler
    target_compile_options(cudaLaunchBench PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:/W4>
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4>
    )
  else()
    # GCC/Clang: standard warnings; for CUDA also forward to host via -Xcompiler
    target_compile_options(cudaLaunchBench PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Wpedantic>
    )
  endif()
endif()

# Helpful note
string(REPLACE ";" " " _archs "${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "CMAKE_CUDA_ARCHITECTURES = ${_archs}")
