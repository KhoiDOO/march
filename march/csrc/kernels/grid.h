#ifndef GRID_H
#define GRID_H

#include <cstdint>
#include <cuda_runtime.h>

template <typename Scalar, typename IndexType>
__global__ void identify_active_voxels_kernel(
    const IndexType* voxels,
    const Scalar* values,
    IndexType n_voxels,
    Scalar iso,
    uint8_t* voxel_codes
) {
    IndexType voxel_idx = (IndexType) blockIdx.x * blockDim.x + threadIdx.x;
    if (voxel_idx >= n_voxels) return;

    IndexType const *v_ptr = &voxels[voxel_idx * 8]; // 8 vertex indices of the voxel
    uint8_t code = 0;
    for (int i = 0; i < 8; ++i) {
        if (values[v_ptr[local_v_idx[i]]] >= iso) {
            code |= (1 << i);
        }
    }
    voxel_codes[voxel_idx] = (uint8_t)code;
};

template <typename IndexType>
__global__ void compact_active_voxels_kernel(
    const uint8_t* voxel_codes,
    const IndexType* prefix_sum,
    IndexType n_voxels,
    IndexType* used_voxel_index,
    uint8_t* used_voxel_code
) {
    IndexType voxel_idx = (IndexType) blockIdx.x * blockDim.x + threadIdx.x;
    if (voxel_idx >= n_voxels) return;

    uint8_t code = voxel_codes[voxel_idx];

    if (code > 0 && code < 255) { // Active voxel

        IndexType pos = prefix_sum[voxel_idx];
        used_voxel_index[pos] = voxel_idx;
        used_voxel_code[pos] = code;
    }
};

#endif // GRID_H

template __global__ void identify_active_voxels_kernel<float, int>(
    const int*, const float*, int, float, uint8_t*);
template __global__ void identify_active_voxels_kernel<float, long long>(
    const long long*, const float*, long long, float, uint8_t*);

template __global__ void compact_active_voxels_kernel<int>(
    const uint8_t*, const int*, int, int*, uint8_t*);
template __global__ void compact_active_voxels_kernel<long long>(
    const uint8_t*, const long long*, long long, long long*, uint8_t*);