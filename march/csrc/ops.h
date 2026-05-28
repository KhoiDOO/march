#ifndef OPS_H
#define OPS_H

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>

inline bool check_cuda_result(cudaError_t code, const char *file, int line)
{
    if (code == cudaSuccess)
        return true;

    fprintf(stderr, "CUDA error %u: %s (%s:%d)\n", unsigned(code), cudaGetErrorString(code), file, line);
    return false;
}

#define CHECK_CUDA(code) check_cuda_result((code), __FILE__, __LINE__)

template <typename T>
inline __device__ __host__ T min(T a, T b) { return a < b ? a : b; }

template <typename T>
inline __device__ __host__ T max(T a, T b) { return a > b ? a : b; }

template <typename T>
inline __device__ __host__ T clamp(T x, T a, T b) { return min(max(a, x), b); }

template <typename T>
inline __device__ __host__ T d_abs(T x) { return x < T(0.0) ? -x : x; }

struct IsActiveOp
{
    __host__ __device__ int operator()(const uint8_t code) const
    {
        return (code > 0 && code < 255) ? 1 : 0;
    }
};

#endif // OPS_H