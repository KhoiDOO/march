#include "grid.h"
#include <thrust/device_ptr.h>
#include <thrust/transform_reduce.h>
#include <thrust/scan.h>
#include <thrust/device_vector.h>

using primitive::Vertex;
using primitive::BBox;

namespace grid {

    template <typename Scalar>
    struct PointToBBox {
        __device__ __host__
        BBox<Scalar> operator()(const Vertex<Scalar>& pt) const {
            return BBox<Scalar>::from_point(pt);
        }
    };

    template <typename Scalar>
    struct BBoxReduce {
        __device__ __host__
        BBox<Scalar> operator()(const BBox<Scalar>& a, const BBox<Scalar>& b) const {
            return a.merge(b);
        }
    };

    template <typename Scalar, typename IndexType>
    __global__ void count_points_in_voxels_kernel(
        const Vertex<Scalar>* points,
        IndexType num_points,
        Scalar min_x, Scalar min_y, Scalar min_z,
        Scalar cx, Scalar cy, Scalar cz,
        IndexType res_x, IndexType res_y, IndexType res_z,
        int* voxel_counts
    ) {
        IndexType idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_points) return;

        Vertex<Scalar> p = points[idx];
        
        IndexType i = min(max((IndexType)((p.x - min_x) / cx), (IndexType)0), res_x - 1);
        IndexType j = min(max((IndexType)((p.y - min_y) / cy), (IndexType)0), res_y - 1);
        IndexType k = min(max((IndexType)((p.z - min_z) / cz), (IndexType)0), res_z - 1);

        IndexType voxel_idx = k * (res_y * res_x) + j * res_x + i;
        atomicAdd(&voxel_counts[voxel_idx], 1);
    }

    template <typename IndexType>
    __global__ void mark_active_vertices_kernel(
        const int* voxel_active_flags,
        IndexType res_x, IndexType res_y, IndexType res_z,
        int* active_vertices
    ) {
        IndexType voxel_idx = blockIdx.x * blockDim.x + threadIdx.x;
        IndexType total_voxels = res_x * res_y * res_z;
        if (voxel_idx >= total_voxels) return;

        if (voxel_active_flags[voxel_idx]) {
            IndexType k = voxel_idx / (res_y * res_x);
            IndexType j = (voxel_idx % (res_y * res_x)) / res_x;
            IndexType i = voxel_idx % res_x;

            IndexType stride_y = res_x + 1;
            IndexType stride_z = (res_y + 1) * (res_x + 1);
            
            // Mark all 8 vertices of this cube
            for(int dz=0; dz<=1; ++dz) {
                for(int dy=0; dy<=1; ++dy) {
                    for(int dx=0; dx<=1; ++dx) {
                        IndexType v_idx = (k + dz) * stride_z + (j + dy) * stride_y + (i + dx);
                        active_vertices[v_idx] = 1;
                    }
                }
            }
        }
    }

    template <typename Scalar, typename IndexType>
    __global__ void generate_sparse_vertices_kernel(
        const int* active_vertices,
        const int* vertex_prefix_sum,
        Vertex<Scalar>* out_vertices,
        Scalar min_x, Scalar min_y, Scalar min_z,
        Scalar cx, Scalar cy, Scalar cz,
        IndexType res_x, IndexType res_y, IndexType res_z
    ) {
        IndexType v_idx = blockIdx.x * blockDim.x + threadIdx.x;
        IndexType total_verts = (res_x + 1) * (res_y + 1) * (res_z + 1);
        if (v_idx >= total_verts) return;

        if (active_vertices[v_idx]) {
            IndexType out_idx = vertex_prefix_sum[v_idx];
            
            IndexType stride_y = res_x + 1;
            IndexType stride_z = (res_y + 1) * (res_x + 1);

            IndexType k = v_idx / stride_z;
            IndexType j = (v_idx % stride_z) / stride_y;
            IndexType i = v_idx % stride_y;

            out_vertices[out_idx] = {
                min_x + i * cx,
                min_y + j * cy,
                min_z + k * cz
            };
        }
    }

    template <typename IndexType>
    __global__ void dilate_voxels_kernel(
        const int* voxel_active_in,
        int* voxel_active_out,
        IndexType res_x, IndexType res_y, IndexType res_z,
        int num_keep
    ) {
        IndexType voxel_idx = blockIdx.x * blockDim.x + threadIdx.x;
        IndexType total_voxels = res_x * res_y * res_z;
        if (voxel_idx >= total_voxels) return;

        IndexType k = voxel_idx / (res_y * res_x);
        IndexType j = (voxel_idx % (res_y * res_x)) / res_x;
        IndexType i = voxel_idx % res_x;

        int active = 0;
        int d_k_start = max(0, (int)k - num_keep);
        int d_k_end = min((int)res_z - 1, (int)k + num_keep);
        int d_j_start = max(0, (int)j - num_keep);
        int d_j_end = min((int)res_y - 1, (int)j + num_keep);
        int d_i_start = max(0, (int)i - num_keep);
        int d_i_end = min((int)res_x - 1, (int)i + num_keep);

        for (int dk = d_k_start; dk <= d_k_end; ++dk) {
            for (int dj = d_j_start; dj <= d_j_end; ++dj) {
                for (int di = d_i_start; di <= d_i_end; ++di) {
                    IndexType neighbor_idx = dk * res_y * res_x + dj * res_x + di;
                    if (voxel_active_in[neighbor_idx]) {
                        active = 1;
                        break;
                    }
                }
                if (active) break;
            }
            if (active) break;
        }

        voxel_active_out[voxel_idx] = active;
    }

    template <typename IndexType>
    __global__ void generate_sparse_cubes_kernel(
        const int* voxel_active_flags,
        const int* cube_prefix_sum,
        const int* vertex_prefix_sum,
        IndexType* out_cubes,
        IndexType res_x, IndexType res_y, IndexType res_z
    ) {
        IndexType voxel_idx = blockIdx.x * blockDim.x + threadIdx.x;
        IndexType total_voxels = res_x * res_y * res_z;
        if (voxel_idx >= total_voxels) return;

        if (voxel_active_flags[voxel_idx]) {
            IndexType out_idx = cube_prefix_sum[voxel_idx];

            IndexType k = voxel_idx / (res_y * res_x);
            IndexType j = (voxel_idx % (res_y * res_x)) / res_x;
            IndexType i = voxel_idx % res_x;

            IndexType stride_y = res_x + 1;
            IndexType stride_z = (res_y + 1) * (res_x + 1);

            IndexType base = out_idx * 8;
            int counter = 0;
            // Iterate in identically the same PyTorch order requested (v0 -> v7 mapping by dx changing fastest)
            for(int dz=0; dz<=1; ++dz) {
                for(int dy=0; dy<=1; ++dy) {
                    for(int dx=0; dx<=1; ++dx) {
                        IndexType v_idx = (k + dz) * stride_z + (j + dy) * stride_y + (i + dx);
                        out_cubes[base + counter] = vertex_prefix_sum[v_idx];
                        counter++;
                    }
                }
            }
        }
    }

    struct ThresholdOp {
        int threshold;
        __host__ __device__ int operator()(int count) const { return count >= threshold ? 1 : 0; }
    };

    template <typename Scalar, typename IndexType>
    void pc_to_voxel_grid(
        Vertex<Scalar> const *points,
        IndexType num_points,
        IndexType res_x, 
        IndexType res_y, 
        IndexType res_z, 
        int k_threshold,
        int num_keep,
        float r,
        Vertex<Scalar>** out_vertices,
        IndexType* out_num_vertices,
        IndexType** out_cubes,
        IndexType* out_num_cubes,
        int device
    ) {
        cudaSetDevice(device);
        if (num_points == 0) return;

        // 1. Get Bounds safely inside GPU
        BBox<Scalar> init_box = {{1e9f, 1e9f, 1e9f}, {-1e9f, -1e9f, -1e9f}};
        thrust::device_ptr<const Vertex<Scalar>> d_pts(points);
        BBox<Scalar> bbox = thrust::transform_reduce(
            d_pts, d_pts + num_points, PointToBBox<Scalar>(), init_box, BBoxReduce<Scalar>());

        Scalar cx = (bbox.max_pt.x - bbox.min_pt.x) / (Scalar)res_x;
        Scalar cy = (bbox.max_pt.y - bbox.min_pt.y) / (Scalar)res_y;
        Scalar cz = (bbox.max_pt.z - bbox.min_pt.z) / (Scalar)res_z;

        IndexType total_voxels = res_x * res_y * res_z;
        IndexType total_dense_verts = (res_x + 1) * (res_y + 1) * (res_z + 1);

        int threads = 256;

        // 2. Count points inside Voxels
        thrust::device_vector<int> voxel_counts(total_voxels, 0);
        int blocks = (num_points + threads - 1) / threads;
        count_points_in_voxels_kernel<<<blocks, threads>>>(
            points, num_points, 
            bbox.min_pt.x, bbox.min_pt.y, bbox.min_pt.z, cx, cy, cz, 
            res_x, res_y, res_z, thrust::raw_pointer_cast(voxel_counts.data())
        );

        // 3. Dilate the active regions
        thrust::device_vector<int> active_voxel_flags_in(total_voxels);
        thrust::transform(voxel_counts.begin(), voxel_counts.end(), active_voxel_flags_in.begin(), ThresholdOp{k_threshold});

        thrust::device_vector<int> active_voxel_flags(total_voxels);
        blocks = (total_voxels + threads - 1) / threads;
        dilate_voxels_kernel<<<blocks, threads>>>(
            thrust::raw_pointer_cast(active_voxel_flags_in.data()),
            thrust::raw_pointer_cast(active_voxel_flags.data()),
            res_x, res_y, res_z, num_keep
        );

        // 4. Mark the active vertices
        thrust::device_vector<int> active_vertices(total_dense_verts, 0);
        mark_active_vertices_kernel<<<blocks, threads>>>(
            thrust::raw_pointer_cast(active_voxel_flags.data()),
            res_x, res_y, res_z, thrust::raw_pointer_cast(active_vertices.data())
        );

        // 5. Cube Memory Prep & Threshold Check
        thrust::device_vector<int> cube_prefix_sums(total_voxels);
        thrust::exclusive_scan(active_voxel_flags.begin(), active_voxel_flags.end(), cube_prefix_sums.begin());

        *out_num_cubes = cube_prefix_sums.back() + active_voxel_flags.back();
        if (*out_num_cubes == 0) return; 

        // 6. Sparse Vertex Check & Output Assignment
        thrust::device_vector<int> vertex_prefix_sums(total_dense_verts);
        thrust::exclusive_scan(active_vertices.begin(), active_vertices.end(), vertex_prefix_sums.begin());
        *out_num_vertices = vertex_prefix_sums.back() + active_vertices.back();

        cudaMalloc(out_vertices, (*out_num_vertices) * sizeof(Vertex<Scalar>));
        cudaMalloc(out_cubes, (*out_num_cubes) * 8 * sizeof(IndexType));

        // 7. Final Generation Kernels Map Values Properly
        int blocks_verts = (total_dense_verts + threads - 1) / threads;
        generate_sparse_vertices_kernel<<<blocks_verts, threads>>>(
            thrust::raw_pointer_cast(active_vertices.data()), thrust::raw_pointer_cast(vertex_prefix_sums.data()),
            *out_vertices, bbox.min_pt.x, bbox.min_pt.y, bbox.min_pt.z,
            cx, cy, cz, res_x, res_y, res_z
        );

        generate_sparse_cubes_kernel<<<blocks, threads>>>(
            thrust::raw_pointer_cast(active_voxel_flags.data()), thrust::raw_pointer_cast(cube_prefix_sums.data()), thrust::raw_pointer_cast(vertex_prefix_sums.data()),
            *out_cubes, res_x, res_y, res_z
        );
        cudaDeviceSynchronize();
    }

    // Explicit template instantiations
    template void pc_to_voxel_grid<float, int>(
        Vertex<float> const *points, int num_points,
        int res_x, int res_y, int res_z, int k_threshold, int num_keep, float r,
        Vertex<float>** out_vertices, int* out_num_vertices,
        int** out_cubes, int* out_num_cubes,
        int device
    );
}