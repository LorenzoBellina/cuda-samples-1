# particles - Particles

## Description

This sample uses CUDA to simulate and visualize a large set of particles and their physical interaction.  Adding "-particles=<N>" to the command line will allow users to set # of particles for simulation.  This example implements a uniform grid data structure using either atomic operations or a fast radix sort from the Thrust library

## Key Concepts

Graphics Interop, Data Parallel Algorithms, Physically-Based Simulation, Performance Strategies

## Supported SM Architectures

[SM 5.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 5.2 ](https://developer.nvidia.com/cuda-gpus)  [SM 5.3 ](https://developer.nvidia.com/cuda-gpus)  [SM 6.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 6.1 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.2 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.5 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.6 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.7 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.9 ](https://developer.nvidia.com/cuda-gpus)  [SM 9.0 ](https://developer.nvidia.com/cuda-gpus)

## Supported OSes

Linux, Windows

## Supported CPU Architecture

x86_64, armv7l

## CUDA APIs involved

### [CUDA Runtime API](http://docs.nvidia.com/cuda/cuda-runtime-api/index.html)
cudaGraphicsUnmapResources, cudaMemcpy, cudaFree, cudaGraphicsResourceGetMappedPointer, cudaGraphicsMapResources, cudaDeviceSynchronize, cudaMemset, cudaMemcpyToSymbol, cudaGraphicsGLRegisterBuffer, cudaGraphicsUnregisterResource, cudaMalloc, cudaInit, cudaGLInit

## Dependencies needed to build/run
[X11](../../../README.md#x11), [OpenGL](../../../README.md#opengl), [Freeglut](../../../README.md#freeglut), [GLEW](../../../README.md#glew)

## Prerequisites

Download and install the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) for your corresponding platform.
Make sure the dependencies mentioned in [Dependencies]() section above are installed.

## References (for more details)

[whitepaper](./doc/particles.pdf)

## Benchmark options (project extensions)

Headless runs use `-benchmark`; the options below are added on top of the original `-n`, `-grid`, `-i`.

| Option | Default | Meaning |
|---|---|---|
| `-mode=<name>` | `base` | collision kernel variant (`base` = original sync-free gather kernel) |
| `-block=<k>` | `64` | threads per block of the collision kernel (multiple of 32) |
| `-warmup=<k>` | `10` | warm-up steps; the system is then reset to the same initial state |
| `-dump=<file>` | | write the final positions (float4 per particle) |
| `-compare=<file>` | | compare the final positions with a dump, exit code 1 if any particle differs more than `-tol` |
| `-tol=<x>` | `1e-3` | per-component position tolerance for `-compare` |
| `-backend=<b>` | `gpu` | `gpu` (CUDA) or `omp` (CPU, OpenMP version of the same algorithm, `particleSystem_omp.cpp`) |
| `-threads=<k>` | all cores | OpenMP threads (same as `OMP_NUM_THREADS`) |
| `-schedule=<kind>[:<chunk>]` | `static` | OpenMP schedule of the collide loop (`static`, `dynamic`, `guided`), which uses `schedule(runtime)` |

Each run prints the mean GPU time of every phase (integrate, hash, sort, reorder, collide), measured with
CUDA events without synchronizing inside the loop, and a `CSV,...` line to collect results with `grep "^CSV"`.

Example: validate a variant against the baseline (use few iterations: the system is chaotic and float
differences grow over time).

```
particles -benchmark -n=65536 -i=50 -dump=base.bin
particles -benchmark -n=65536 -i=50 -mode=<variant> -compare=base.bin
```

The OpenMP backend gives bit-identical results for any number of threads and schedule, and matches the GPU
within rounding (max position difference ~1e-5 after 25 steps, ~1e-3 after 50 because of the chaotic dynamics).
`scaling_omp.ps1` runs the strong and weak scaling series and writes them to a CSV file.
On Windows the OpenMP backend is built with `/openmp:llvm` (OpenMP 3.x runtime, needed by `omp_set_schedule`);
CMake copies `libomp140.x86_64.dll` next to the executable.
