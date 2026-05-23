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

    template <typename IndexType>
    struct ThresholdOp {
        int threshold;
        __host__ __device__ uint8_t operator()(IndexType count) const { return count >= (IndexType)threshold ? 1 : 0; }
    };

    template <typename Scalar, typename IndexType>
    __global__ void count_points_in_voxels_kernel(
        const Vertex<Scalar>* points,
        IndexType num_points,
        Scalar min_x, Scalar min_y, Scalar min_z,
        Scalar cx, Scalar cy, Scalar cz,
        IndexType res_x, IndexType res_y, IndexType res_z,
        IndexType* voxel_counts
    ) {
        IndexType idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_points) return;

        Vertex<Scalar> p = points[idx];

        Scalar fi = (p.x - min_x) / cx;
        Scalar fj = (p.y - min_y) / cy;
        Scalar fk = (p.z - min_z) / cz;

        if (fi < 0 || fi >= (Scalar)res_x ||
            fj < 0 || fj >= (Scalar)res_y ||
            fk < 0 || fk >= (Scalar)res_z) {
                return;
            }
        
        IndexType i = min(max((IndexType)fi, (IndexType)0), res_x - 1);
        IndexType j = min(max((IndexType)fj, (IndexType)0), res_y - 1);
        IndexType k = min(max((IndexType)fk, (IndexType)0), res_z - 1);

        IndexType voxel_idx = k * (res_y * res_x) + j * res_x + i;
        if constexpr (sizeof(IndexType) == 8) {
            atomicAdd((unsigned long long*)&voxel_counts[voxel_idx], 1ULL);
        } else {
            atomicAdd((unsigned int*)&voxel_counts[voxel_idx], 1U);
        }
    }

    template <typename Scalar, typename IndexType>
    __global__ void count_points_chunk_kernel(
        const Vertex<Scalar>* points,
        IndexType num_points,
        Scalar min_x, Scalar min_y, Scalar min_z,
        Scalar cx, Scalar cy, Scalar cz,
        IndexType global_res_x, IndexType global_res_y, IndexType global_res_z,
        IndexType start_x, IndexType start_y, IndexType start_z,
        IndexType p_res_x, IndexType p_res_y, IndexType p_res_z,
        IndexType* voxel_counts 
    ) {
        IndexType idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_points) return;

        Vertex<Scalar> p = points[idx];
        Scalar fi = (p.x - min_x) / cx;
        Scalar fj = (p.y - min_y) / cy;
        Scalar fk = (p.z - min_z) / cz;

        // Discard if outside global bounding box
        if (fi < 0 || fi >= (Scalar)global_res_x ||
            fj < 0 || fj >= (Scalar)global_res_y ||
            fk < 0 || fk >= (Scalar)global_res_z) return;

        IndexType i = (IndexType)fi;
        IndexType j = (IndexType)fj;
        IndexType k = (IndexType)fk;

        // Check if point falls inside this specific padded chunk
        if (i >= start_x && i < start_x + p_res_x &&
            j >= start_y && j < start_y + p_res_y &&
            k >= start_z && k < start_z + p_res_z) {
            
            IndexType li = i - start_x;
            IndexType lj = j - start_y;
            IndexType lk = k - start_z;
            IndexType voxel_idx = lk * (p_res_y * p_res_x) + lj * p_res_x + li;

            if constexpr (sizeof(IndexType) == 8) {
                atomicAdd((unsigned long long*)&voxel_counts[voxel_idx], 1ULL);
            } else {
                atomicAdd((unsigned int*)&voxel_counts[voxel_idx], 1U);
            }
        }
    }

    template <typename IndexType>
    __global__ void filter_core_voxels_kernel(
        uint8_t* active_voxel_flags,
        IndexType p_res_x, IndexType p_res_y, IndexType p_res_z,
        IndexType core_start_x, IndexType core_start_y, IndexType core_start_z,
        IndexType core_end_x, IndexType core_end_y, IndexType core_end_z
    ) {
        IndexType voxel_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (voxel_idx >= p_res_x * p_res_y * p_res_z) return;

        IndexType k = voxel_idx / (p_res_y * p_res_x);
        IndexType j = (voxel_idx % (p_res_y * p_res_x)) / p_res_x;
        IndexType i = voxel_idx % p_res_x;

        // Turn off active flags for the padded boundary so we don't generate duplicate cubes
        if (i < core_start_x || i >= core_end_x ||
            j < core_start_y || j >= core_end_y ||
            k < core_start_z || k >= core_end_z) {
            active_voxel_flags[voxel_idx] = 0;
        }
    }

    template <typename IndexType>
    __global__ void mark_active_vertices_kernel(
        const uint8_t* voxel_active_flags,
        IndexType res_x, IndexType res_y, IndexType res_z,
        uint8_t* active_vertices
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
        const uint8_t* active_vertices,
        const IndexType* vertex_prefix_sum,
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
        const uint8_t* voxel_active_in,
        uint8_t* voxel_active_out,
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
        const uint8_t* voxel_active_flags,
        const IndexType* cube_prefix_sum,
        const IndexType* vertex_prefix_sum,
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
            
            // We still need the loops to visit all 8 corners of the cube
            for(int dz=0; dz<=1; ++dz) {
                for(int dy=0; dy<=1; ++dy) {
                    for(int dx=0; dx<=1; ++dx) {
                        IndexType v_idx = (k + dz) * stride_z + (j + dy) * stride_y + (i + dx);
                        
                        int u = dx;  // Bit 0: Left-to-right axis
                        int v = dz;      // Bit 1: Back-to-front is now Z
                        int w = dy;      // Bit 2: Upward is now Y
                        
                        int topo_idx = (w << 2) | (v << 1) | u;
                        // int topo_idx = dz * 4 + dy * 2 + dx;
                        
                        out_cubes[base + topo_idx] = vertex_prefix_sum[v_idx];
                    }
                }
            }
        }
    }

    template <typename Scalar, typename IndexType>
    void pc_to_voxel_grid(
        Vertex<Scalar> const *points,
        IndexType num_points,
        IndexType res_x, 
        IndexType res_y, 
        IndexType res_z, 
        int k_threshold,
        int num_keep,
        float rmi_x,
        float rmi_y,
        float rmi_z,
        float rma_x,
        float rma_y,
        float rma_z,
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
        
        // Apply the truncation factor 'r' to the bounding box
        bbox.min_pt.x *= rmi_x;
        bbox.min_pt.y *= rmi_y;
        bbox.min_pt.z *= rmi_z;
        bbox.max_pt.x *= rma_x;
        bbox.max_pt.y *= rma_y;
        bbox.max_pt.z *= rma_z;

        Scalar cx = (bbox.max_pt.x - bbox.min_pt.x) / (Scalar)res_x;
        Scalar cy = (bbox.max_pt.y - bbox.min_pt.y) / (Scalar)res_y;
        Scalar cz = (bbox.max_pt.z - bbox.min_pt.z) / (Scalar)res_z;

        IndexType total_voxels = res_x * res_y * res_z;
        IndexType total_dense_verts = (res_x + 1) * (res_y + 1) * (res_z + 1);

        IndexType threads = 256;

        // 2. Count points inside Voxels
        thrust::device_vector<IndexType> voxel_counts_and_prefix(total_voxels, 0);
        IndexType blocks = (num_points + threads - 1) / threads;
        count_points_in_voxels_kernel<<<blocks, threads>>>(
            points, num_points, 
            bbox.min_pt.x, bbox.min_pt.y, bbox.min_pt.z, cx, cy, cz, 
            res_x, res_y, res_z, thrust::raw_pointer_cast(voxel_counts_and_prefix.data())
        );

        // 3. Dilate the active regions
        thrust::device_vector<uint8_t> active_voxel_flags(total_voxels, 0);
        {
            thrust::device_vector<uint8_t> active_voxel_flags_in(total_voxels, 0);
            thrust::transform(
                voxel_counts_and_prefix.begin(), voxel_counts_and_prefix.end(), 
                active_voxel_flags_in.begin(), 
                ThresholdOp<IndexType>{k_threshold}
            );

            IndexType dilate_blocks = (total_voxels + threads - 1) / threads;
            dilate_voxels_kernel<<<dilate_blocks, threads>>>(
                thrust::raw_pointer_cast(active_voxel_flags_in.data()),
                thrust::raw_pointer_cast(active_voxel_flags.data()),
                res_x, res_y, res_z, num_keep
            );
        }

        // 4. Mark the active vertices
        thrust::device_vector<uint8_t> active_vertices(total_dense_verts, 0);
        IndexType mark_blocks = (total_voxels + threads - 1) / threads;
        mark_active_vertices_kernel<<<mark_blocks, threads>>>(
            thrust::raw_pointer_cast(active_voxel_flags.data()),
            res_x, res_y, res_z, thrust::raw_pointer_cast(active_vertices.data())
        );

        // 5. Cube Memory Prep & Threshold Check
        thrust::exclusive_scan(active_voxel_flags.begin(), active_voxel_flags.end(), voxel_counts_and_prefix.begin(), (IndexType)0);

        *out_num_cubes = voxel_counts_and_prefix.back() + active_voxel_flags.back();
        if (*out_num_cubes == 0) return; 

        // 6. Sparse Vertex Check & Output Assignment
        thrust::device_vector<IndexType> vertex_prefix_sums(total_dense_verts, 0);
        thrust::exclusive_scan(
            active_vertices.begin(), 
            active_vertices.end(), 
            vertex_prefix_sums.begin(), 
            (IndexType)0
        );
        *out_num_vertices = vertex_prefix_sums.back() + active_vertices.back();

        cudaMalloc(out_vertices, (*out_num_vertices) * sizeof(Vertex<Scalar>));
        cudaMalloc(out_cubes, (*out_num_cubes) * 8 * sizeof(IndexType));

        // 7. Final Generation Kernels Map Values Properly
        IndexType blocks_verts = (total_dense_verts + threads - 1) / threads;
        generate_sparse_vertices_kernel<<<blocks_verts, threads>>>(
            thrust::raw_pointer_cast(active_vertices.data()), thrust::raw_pointer_cast(vertex_prefix_sums.data()),
            *out_vertices, bbox.min_pt.x, bbox.min_pt.y, bbox.min_pt.z,
            cx, cy, cz, res_x, res_y, res_z
        );

        IndexType blocks_cubes = (total_voxels + threads - 1) / threads;
        generate_sparse_cubes_kernel<<<blocks_cubes, threads>>>(
            thrust::raw_pointer_cast(active_voxel_flags.data()), 
            thrust::raw_pointer_cast(voxel_counts_and_prefix.data()), 
            thrust::raw_pointer_cast(vertex_prefix_sums.data()),
            *out_cubes, res_x, res_y, res_z
        );
        cudaDeviceSynchronize();
    }

    template <typename Scalar, typename IndexType>
    void pc_to_voxel_grid_chunk(
        Vertex<Scalar> const *points,
        IndexType num_points,
        IndexType res_x, IndexType res_y, IndexType res_z, 
        int chunk_size,
        int k_threshold, int num_keep,
        float rmi_x, float rmi_y, float rmi_z,
        float rma_x, float rma_y, float rma_z,
        Vertex<Scalar>** out_vertices, IndexType* out_num_vertices,
        IndexType** out_cubes, IndexType* out_num_cubes,
        int device
    ) {
        cudaSetDevice(device);
        if (num_points == 0) return;

        BBox<Scalar> init_box = {{1e9f, 1e9f, 1e9f}, {-1e9f, -1e9f, -1e9f}};
        thrust::device_ptr<const Vertex<Scalar>> d_pts(points);
        BBox<Scalar> bbox = thrust::transform_reduce(
            d_pts, d_pts + num_points, PointToBBox<Scalar>(), init_box, BBoxReduce<Scalar>());
        
        bbox.min_pt.x *= rmi_x; bbox.min_pt.y *= rmi_y; bbox.min_pt.z *= rmi_z;
        bbox.max_pt.x *= rma_x; bbox.max_pt.y *= rma_y; bbox.max_pt.z *= rma_z;

        Scalar cx = (bbox.max_pt.x - bbox.min_pt.x) / (Scalar)res_x;
        Scalar cy = (bbox.max_pt.y - bbox.min_pt.y) / (Scalar)res_y;
        Scalar cz = (bbox.max_pt.z - bbox.min_pt.z) / (Scalar)res_z;

        // Host accumulation arrays (Memory is bounded by actual geometry, not grid resolution)
        std::vector<Vertex<Scalar>> host_all_vertices;
        std::vector<IndexType> host_all_cubes;
        IndexType total_verts_so_far = 0;
        IndexType threads = 256;

        for (IndexType sz = 0; sz < res_z; sz += chunk_size) {
            for (IndexType sy = 0; sy < res_y; sy += chunk_size) {
                for (IndexType sx = 0; sx < res_x; sx += chunk_size) {
                    
                    // 1. Calculate Chunk Bounds with Padding
                    IndexType ex = min(res_x, sx + chunk_size);
                    IndexType ey = min(res_y, sy + chunk_size);
                    IndexType ez = min(res_z, sz + chunk_size);

                    IndexType p_sx = max((IndexType)0, sx - num_keep);
                    IndexType p_sy = max((IndexType)0, sy - num_keep);
                    IndexType p_sz = max((IndexType)0, sz - num_keep);

                    IndexType p_ex = min(res_x, ex + num_keep);
                    IndexType p_ey = min(res_y, ey + num_keep);
                    IndexType p_ez = min(res_z, ez + num_keep);

                    IndexType p_res_x = p_ex - p_sx;
                    IndexType p_res_y = p_ey - p_sy;
                    IndexType p_res_z = p_ez - p_sz;

                    IndexType total_voxels = p_res_x * p_res_y * p_res_z;
                    IndexType total_dense_verts = (p_res_x + 1) * (p_res_y + 1) * (p_res_z + 1);

                    // 2. Count Points
                    thrust::device_vector<IndexType> voxel_counts(total_voxels, 0);
                    IndexType blocks_pts = (num_points + threads - 1) / threads;
                    count_points_chunk_kernel<<<blocks_pts, threads>>>(
                        points, num_points, 
                        bbox.min_pt.x, bbox.min_pt.y, bbox.min_pt.z, cx, cy, cz,
                        res_x, res_y, res_z,
                        p_sx, p_sy, p_sz, p_res_x, p_res_y, p_res_z,
                        thrust::raw_pointer_cast(voxel_counts.data())
                    );

                    // 3. Dilate
                    thrust::device_vector<uint8_t> active_voxel_flags(total_voxels, 0);
                    {
                        thrust::device_vector<uint8_t> active_voxel_flags_in(total_voxels, 0);
                        thrust::transform(
                            voxel_counts.begin(), voxel_counts.end(), 
                            active_voxel_flags_in.begin(), 
                            ThresholdOp<IndexType>{k_threshold}
                        );

                        IndexType dilate_blocks = (total_voxels + threads - 1) / threads;
                        dilate_voxels_kernel<<<dilate_blocks, threads>>>(
                            thrust::raw_pointer_cast(active_voxel_flags_in.data()),
                            thrust::raw_pointer_cast(active_voxel_flags.data()),
                            p_res_x, p_res_y, p_res_z, num_keep
                        );
                    }
                    
                    // 4. Filter out Padded Bounds (Prevents duplicate cubes between chunks)
                    IndexType filter_blocks = (total_voxels + threads - 1) / threads;
                    filter_core_voxels_kernel<<<filter_blocks, threads>>>(
                        thrust::raw_pointer_cast(active_voxel_flags.data()),
                        p_res_x, p_res_y, p_res_z,
                        sx - p_sx, sy - p_sy, sz - p_sz, 
                        ex - p_sx, ey - p_sy, ez - p_sz
                    );

                    // 5. Mark Vertices
                    thrust::device_vector<uint8_t> active_vertices(total_dense_verts, 0);
                    IndexType mark_blocks = (total_voxels + threads - 1) / threads;
                    mark_active_vertices_kernel<<<mark_blocks, threads>>>(
                        thrust::raw_pointer_cast(active_voxel_flags.data()),
                        p_res_x, p_res_y, p_res_z, thrust::raw_pointer_cast(active_vertices.data())
                    );

                    // 6. Prefix Sums
                    thrust::device_vector<IndexType> cube_prefix_sums(total_voxels, 0);
                    thrust::exclusive_scan(active_voxel_flags.begin(), active_voxel_flags.end(), cube_prefix_sums.begin(), (IndexType)0);
                    IndexType chunk_num_cubes = cube_prefix_sums.back() + (IndexType)active_voxel_flags.back();

                    if (chunk_num_cubes == 0) continue;

                    thrust::device_vector<IndexType> vertex_prefix_sums(total_dense_verts, 0);
                    thrust::exclusive_scan(active_vertices.begin(), active_vertices.end(), vertex_prefix_sums.begin(), (IndexType)0);
                    IndexType chunk_num_vertices = vertex_prefix_sums.back() + (IndexType)active_vertices.back();

                    // 7. Extract Geometry
                    thrust::device_vector<Vertex<Scalar>> d_chunk_verts(chunk_num_vertices);
                    thrust::device_vector<IndexType> d_chunk_cubes(chunk_num_cubes * 8);

                    IndexType blocks_verts = (total_dense_verts + threads - 1) / threads;
                    generate_sparse_vertices_kernel<<<blocks_verts, threads>>>(
                        thrust::raw_pointer_cast(active_vertices.data()), 
                        thrust::raw_pointer_cast(vertex_prefix_sums.data()),
                        thrust::raw_pointer_cast(d_chunk_verts.data()), 
                        bbox.min_pt.x + p_sx * cx, bbox.min_pt.y + p_sy * cy, bbox.min_pt.z + p_sz * cz,
                        cx, cy, cz, p_res_x, p_res_y, p_res_z
                    );

                    IndexType blocks_cubes = (total_voxels + threads - 1) / threads;
                    generate_sparse_cubes_kernel<<<blocks_cubes, threads>>>(
                        thrust::raw_pointer_cast(active_voxel_flags.data()), 
                        thrust::raw_pointer_cast(cube_prefix_sums.data()), 
                        thrust::raw_pointer_cast(vertex_prefix_sums.data()),
                        thrust::raw_pointer_cast(d_chunk_cubes.data()), 
                        p_res_x, p_res_y, p_res_z
                    );

                    // 8. Accumulate on Host
                    std::vector<Vertex<Scalar>> h_chunk_verts(chunk_num_vertices);
                    std::vector<IndexType> h_chunk_cubes(chunk_num_cubes * 8);
                    thrust::copy(d_chunk_verts.begin(), d_chunk_verts.end(), h_chunk_verts.begin());
                    thrust::copy(d_chunk_cubes.begin(), d_chunk_cubes.end(), h_chunk_cubes.begin());

                    host_all_vertices.insert(host_all_vertices.end(), h_chunk_verts.begin(), h_chunk_verts.end());
                    
                    // Offset cube corner indices to point to the correct position in the global array
                    for(auto& idx : h_chunk_cubes) {
                        host_all_cubes.push_back(idx + total_verts_so_far);
                    }
                    total_verts_so_far += chunk_num_vertices;
                }
            }
        }

        // 9. Push final dense result back to GPU
        *out_num_vertices = host_all_vertices.size();
        *out_num_cubes = host_all_cubes.size() / 8;

        if (*out_num_vertices > 0) {
            cudaMalloc(out_vertices, (*out_num_vertices) * sizeof(Vertex<Scalar>));
            cudaMalloc(out_cubes, (*out_num_cubes) * 8 * sizeof(IndexType));
            cudaMemcpy(*out_vertices, host_all_vertices.data(), (*out_num_vertices) * sizeof(Vertex<Scalar>), cudaMemcpyHostToDevice);
            cudaMemcpy(*out_cubes, host_all_cubes.data(), (*out_num_cubes) * 8 * sizeof(IndexType), cudaMemcpyHostToDevice);
        }
    }

    // Explicit template instantiations
    template void pc_to_voxel_grid<float, int>(
        Vertex<float> const *points, int num_points,
        int res_x, int res_y, int res_z, int k_threshold, int num_keep, 
        float rmi_x, float rmi_y, float rmi_z, float rma_x, float rma_y, float rma_z,
        Vertex<float>** out_vertices, int* out_num_vertices,
        int** out_cubes, int* out_num_cubes,
        int device
    );

    template void pc_to_voxel_grid<float, long long>(
        Vertex<float> const *points, long long num_points,
        long long res_x, long long res_y, long long res_z, int k_threshold, int num_keep, 
        float rmi_x, float rmi_y, float rmi_z, float rma_x, float rma_y, float rma_z,
        Vertex<float>** out_vertices, long long* out_num_vertices,
        long long** out_cubes, long long* out_num_cubes,
        int device
    );

    template void pc_to_voxel_grid_chunk<float, int>(
        Vertex<float> const *points, int num_points,
        int res_x, int res_y, int res_z, int chunk_size, int k_threshold, int num_keep, 
        float rmi_x, float rmi_y, float rmi_z, float rma_x, float rma_y, float rma_z,
        Vertex<float>** out_vertices, int* out_num_vertices,
        int** out_cubes, int* out_num_cubes,
        int device
    );

    template void pc_to_voxel_grid_chunk<float, long long>(
        Vertex<float> const *points, long long num_points,
        long long res_x, long long res_y, long long res_z, int chunk_size, int k_threshold, int num_keep, 
        float rmi_x, float rmi_y, float rmi_z, float rma_x, float rma_y, float rma_z,
        Vertex<float>** out_vertices, long long* out_num_vertices,
        long long** out_cubes, long long* out_num_cubes,
        int device
    );
}