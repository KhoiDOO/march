#include "dmc.h"
#include "primitive.h"
#include "constants/dmc_tables.h"
#include "ops.h"

#include <cstdint>
#include <cstdio>

#include <cuda_runtime.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/unique.h>
#include <thrust/binary_search.h>
#include <thrust/iterator/transform_iterator.h>

namespace dmc
{
    template <typename Scalar, typename IndexType>
    void DMC<Scalar, IndexType>::ensure_grid_storage_size(size_t n_voxels)
    {
        if (n_voxels > this->allocated_voxel_count)
        {
            this->allocated_voxel_count = n_voxels + n_voxels / 5; // Add 20% buffer to avoid frequent reallocations

            // Free old memory
            if (this->temp_buffer)
                CHECK_CUDA(cudaFree(this->temp_buffer));
            if (this->voxel_codes)
                CHECK_CUDA(cudaFree(this->voxel_codes));

            // Allocate new memory
            CHECK_CUDA(cudaMalloc((void **)&this->temp_buffer, this->allocated_voxel_count * sizeof(IndexType)));
            CHECK_CUDA(cudaMalloc((void **)&this->voxel_codes, this->allocated_voxel_count * sizeof(uint8_t)));
        }
    };

    template <typename Scalar, typename IndexType>
    void DMC<Scalar, IndexType>::ensure_used_voxel_storage_size(size_t n_used_voxels)
    {
        if (n_used_voxels > this->allocated_used_voxel_count)
        {

            this->allocated_used_voxel_count = n_used_voxels + n_used_voxels / 5; // Add 20% buffer to avoid frequent reallocations

            // Free old memory
            if (this->used_voxel_code)
                CHECK_CUDA(cudaFree(this->used_voxel_code));
            if (this->used_voxel_index)
                CHECK_CUDA(cudaFree(this->used_voxel_index));
            if (this->voxel_edge_to_vert_idx)
                CHECK_CUDA(cudaFree(this->voxel_edge_to_vert_idx));

            // Allocate new memory
            CHECK_CUDA(cudaMalloc((void **)&this->used_voxel_code, this->allocated_used_voxel_count * sizeof(uint8_t)));
            CHECK_CUDA(cudaMalloc((void **)&this->used_voxel_index, this->allocated_used_voxel_count * sizeof(IndexType)));
            CHECK_CUDA(cudaMalloc((void **)&this->voxel_edge_to_vert_idx, this->allocated_used_voxel_count * 12 * sizeof(IndexType)));
        }
    };

    template <typename Scalar, typename IndexType>
    void DMC<Scalar, IndexType>::ensure_edge_storage_size(size_t n_edges, bool with_colors)
    {
        bool missing_color_buffer = (with_colors && this->edge_zero_colors == nullptr);

        if (n_edges > this->allocated_edge_count || missing_color_buffer)
        {
            this->allocated_edge_count = n_edges + n_edges / 5;

            if (this->unique_edges)
                CHECK_CUDA(cudaFree(this->unique_edges));
            if (this->edge_zero_crossings)
                CHECK_CUDA(cudaFree(this->edge_zero_crossings));
            if (this->edge_zero_colors)
                CHECK_CUDA(cudaFree(this->edge_zero_colors));

            CHECK_CUDA(cudaMalloc((void **)&this->unique_edges, this->allocated_edge_count * sizeof(EdgeKey<IndexType>)));
            CHECK_CUDA(cudaMalloc((void **)&this->edge_zero_crossings, this->allocated_edge_count * sizeof(Vertex<Scalar>)));

            if (with_colors)
            {
                CHECK_CUDA(cudaMalloc((void **)&this->edge_zero_colors, this->allocated_edge_count * sizeof(Vertex<Scalar>)));
            }
            else
            {
                this->edge_zero_colors = nullptr;
            }
        }
    }

    template <typename Scalar, typename IndexType>
    void DMC<Scalar, IndexType>::ensure_vert_storage_size(size_t n_dual_verts, bool with_colors)
    {
        bool missing_color_buffer = (with_colors && this->out_colors == nullptr);

        if (n_dual_verts > this->allocated_vert_count || missing_color_buffer)
        {
            if (n_dual_verts > this->allocated_vert_count)
            {
                this->allocated_vert_count = n_dual_verts + n_dual_verts / 5;
            }

            // Free old memory
            if (this->verts)
                CHECK_CUDA(cudaFree(this->verts));
            if (this->out_colors && with_colors)
                CHECK_CUDA(cudaFree(this->out_colors));

            // Allocate new memory
            CHECK_CUDA(cudaMalloc((void **)&this->verts, this->allocated_vert_count * sizeof(Vertex<Scalar>)));
            if (with_colors)
            {
                CHECK_CUDA(cudaMalloc((void **)&this->out_colors, this->allocated_vert_count * sizeof(Vertex<Scalar>)));
            }
            else
            {
                this->out_colors = nullptr;
            }
        }
    }

    template <typename Scalar, typename IndexType>
    void DMC<Scalar, IndexType>::ensure_quad_storage_size(size_t n_quads)
    {
        if (n_quads > this->allocated_quad_count)
        {
            // Add 20% buffer to avoid frequent reallocations
            this->allocated_quad_count = n_quads + n_quads / 5;

            // Free old memory
            if (this->quads)
                CHECK_CUDA(cudaFree(this->quads));

            size_t bytes = this->allocated_quad_count * 4 * sizeof(IndexType);
            CHECK_CUDA(cudaMalloc((void **)&this->quads, bytes));
            CHECK_CUDA(cudaMemset(this->quads, 0, bytes));
        }
        if (n_quads > 0)
        {
            size_t active_bytes = n_quads * 4 * sizeof(IndexType);
            CHECK_CUDA(cudaMemset(this->quads, 0xFF, active_bytes));
        }
    }

    template <typename Scalar, typename IndexType>
    __global__ void identify_active_voxels_kernel(
        const IndexType *voxels,
        const Scalar *values,
        IndexType n_voxels,
        Scalar iso,
        uint8_t *voxel_codes)
    {
        IndexType voxel_idx = (IndexType)blockIdx.x * blockDim.x + threadIdx.x;
        if (voxel_idx >= n_voxels)
            return;

        IndexType const *v_ptr = &voxels[voxel_idx * 8]; // 8 vertex indices of the voxel
        uint8_t code = 0;
        for (int i = 0; i < 8; ++i)
        {
            if (values[v_ptr[i]] < iso)
            {
                code |= (1 << i);
            }
        }
        voxel_codes[voxel_idx] = (uint8_t)code;

        // int to_check = check_table[code][0];
        
        // if (to_check == 1) 
        // {
        //     int dx = check_table[code][1];
        //     int dy = check_table[code][2];
        //     int dz = check_table[code][3];
        //     uint8_t inverted_case = check_table[code][4];

        //     // 1. Extract the 4 vertices of the ambiguous face based on the direction.
        //     // Using Morton order (z<<2 | y<<1 | x)
        //     Scalar face_avg = 0.0;
            
        //     if (dx == 1)  // +X face (x=1)
        //         face_avg = (values[v_ptr[1]] + values[v_ptr[3]] + values[v_ptr[5]] + values[v_ptr[7]]) / 4.0;
        //     else if (dx == -1) // -X face (x=0)
        //         face_avg = (values[v_ptr[0]] + values[v_ptr[2]] + values[v_ptr[4]] + values[v_ptr[6]]) / 4.0;
        //     else if (dy == 1)  // +Y face (y=1)
        //         face_avg = (values[v_ptr[2]] + values[v_ptr[3]] + values[v_ptr[6]] + values[v_ptr[7]]) / 4.0;
        //     else if (dy == -1) // -Y face (y=0)
        //         face_avg = (values[v_ptr[0]] + values[v_ptr[1]] + values[v_ptr[4]] + values[v_ptr[5]]) / 4.0;
        //     else if (dz == 1)  // +Z face (z=1)
        //         face_avg = (values[v_ptr[4]] + values[v_ptr[5]] + values[v_ptr[6]] + values[v_ptr[7]]) / 4.0;
        //     else if (dz == -1) // -Z face (z=0)
        //         face_avg = (values[v_ptr[0]] + values[v_ptr[1]] + values[v_ptr[2]] + values[v_ptr[3]]) / 4.0;

        //     // 2. Both sharing voxels will compute this EXACT same average!
        //     // If the saddle center is greater than the isosurface, we invert.
        //     // This synchronously seals the hole from both sides.
        //     if (face_avg > iso) {
        //         code = inverted_case;
        //     }
        // }

        // voxel_codes[voxel_idx] = code;
    };

    template <typename IndexType>
    __global__ void compact_active_voxels_kernel(
        const uint8_t *voxel_codes,
        const IndexType *prefix_sum,
        IndexType n_voxels,
        IndexType *used_voxel_index,
        uint8_t *used_voxel_code)
    {
        IndexType voxel_idx = (IndexType)blockIdx.x * blockDim.x + threadIdx.x;
        if (voxel_idx >= n_voxels)
            return;

        uint8_t code = voxel_codes[voxel_idx];

        if (code > 0 && code < 255)
        { // Active voxel

            IndexType pos = prefix_sum[voxel_idx];
            used_voxel_index[pos] = voxel_idx;
            used_voxel_code[pos] = code;
        }
    };

    template <typename IndexType>
    __global__ void extract_active_edges_kernel(
        const IndexType *__restrict__ voxels,
        const IndexType *__restrict__ used_voxel_index,
        const uint8_t *__restrict__ used_voxel_code,
        IndexType n_used_voxels,
        EdgeKey<IndexType> *__restrict__ out_edges)
    {
        IndexType idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n_used_voxels)
            return;

        IndexType global_voxel_idx = used_voxel_index[idx];
        uint8_t code = used_voxel_code[idx];

        // Base pointer to the 8 global vertex indices of this specific voxel
        const IndexType *v_ptr = &voxels[global_voxel_idx * 8];

        for (int e = 0; e < 12; ++e)
        {
            int local_v0 = c_edge_v0[e];
            int local_v1 = c_edge_v1[e];

            // Check if the edge crosses the surface by comparing the bits in the voxel code
            bool v0_inside = (code & (1 << local_v0)) != 0;
            bool v1_inside = (code & (1 << local_v1)) != 0;

            if (v0_inside != v1_inside)
            {
                // Active Edge! Fetch the global indices for the grid corners
                IndexType global_v0 = v_ptr[local_v0];
                IndexType global_v1 = v_ptr[local_v1];

                // Ensure consistent ordering so neighboring voxels produce the EXACT same key
                IndexType min_v = min(global_v0, global_v1);
                IndexType max_v = max(global_v0, global_v1);

                out_edges[idx * 12 + e] = EdgeKey<IndexType>{min_v, max_v};
            }
            else
            {
                // Inactive Edge - Write Sentinel value (max possible integer)
                out_edges[idx * 12 + e] = EdgeKey<IndexType>{(IndexType)-1, (IndexType)-1};
            }
        }
    }

    template <typename Scalar, typename IndexType>
    __global__ void compute_edge_zero_crossings_kernel(
        const EdgeKey<IndexType> *__restrict__ unique_edges,
        const Vertex<Scalar> *__restrict__ grid_vertices,
        const Vertex<Scalar> *__restrict__ grid_colors,
        const Scalar *__restrict__ values,
        Scalar iso,
        IndexType n_unique_edges,
        Vertex<Scalar> *__restrict__ edge_zero_crossings,
        Vertex<Scalar> *__restrict__ edge_zero_colors,
        bool with_colors)
    {
        IndexType idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n_unique_edges)
            return;

        EdgeKey<IndexType> edge = unique_edges[idx];
        IndexType v0_idx = edge.v0;
        IndexType v1_idx = edge.v1;

        Vertex<Scalar> p0 = grid_vertices[v0_idx];
        Vertex<Scalar> p1 = grid_vertices[v1_idx];
        Scalar val0 = values[v0_idx];
        Scalar val1 = values[v1_idx];

        Vertex<Scalar> c0, c1;
        if (with_colors)
        {
            c0 = grid_colors[v0_idx];
            c1 = grid_colors[v1_idx];
        }

        const Scalar EPS = Scalar(1e-5);
        Vertex<Scalar> p;
        Vertex<Scalar> c;

        if (d_abs(iso - val0) < EPS)
        {
            p = p0;
            if (with_colors)
                c = c0;
        }
        else if (d_abs(iso - val1) < EPS)
        {
            p = p1;
            if (with_colors)
                c = c1;
        }
        else if (d_abs(val0 - val1) < EPS)
        {
            p = p0;
            if (with_colors)
                c = c0;
        }
        else
        {
            Scalar t = (val1 != val0) ? clamp((iso - val0) / (val1 - val0), Scalar(0.0), Scalar(1.0)) : Scalar(0.5);
            // Assuming your Vertex<Scalar> struct in primitive.h has overloaded + and * operators
            p = p0 + (p1 - p0) * t;
            if (with_colors)
                c = c0 + (c1 - c0) * t;
        }

        edge_zero_crossings[idx] = p;
        if (with_colors)
            edge_zero_colors[idx] = c;
    }

    template <typename IndexType>
    __global__ void map_voxel_edges_to_unique_idx_kernel(
        const IndexType *__restrict__ voxels,
        const IndexType *__restrict__ used_voxel_index,
        const uint8_t *__restrict__ used_voxel_code,
        const EdgeKey<IndexType> *__restrict__ unique_edges,
        IndexType n_used_voxels,
        IndexType n_unique_edges,
        IndexType *__restrict__ voxel_edge_to_vert_idx)
    {
        IndexType idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n_used_voxels)
            return;

        IndexType global_voxel_idx = used_voxel_index[idx];
        uint8_t code = used_voxel_code[idx];
        const IndexType *v_ptr = &voxels[global_voxel_idx * 8];

        for (int e = 0; e < 12; ++e)
        {
            int local_v0 = c_edge_v0[e];
            int local_v1 = c_edge_v1[e];

            bool v0_inside = (code & (1 << local_v0)) != 0;
            bool v1_inside = (code & (1 << local_v1)) != 0;

            if (v0_inside != v1_inside)
            {
                // Reconstruct the EdgeKey
                IndexType global_v0 = v_ptr[local_v0];
                IndexType global_v1 = v_ptr[local_v1];
                IndexType min_v = min(global_v0, global_v1);
                IndexType max_v = max(global_v0, global_v1);
                EdgeKey<IndexType> target_key = {min_v, max_v};

                // Binary search for this key (Utilizing your overloaded < and == operators)
                IndexType left = 0;
                IndexType right = n_unique_edges - 1;
                IndexType found_idx = -1;

                while (left <= right)
                {
                    IndexType mid = left + (right - left) / 2;
                    EdgeKey<IndexType> mid_key = unique_edges[mid];

                    if (mid_key == target_key)
                    {
                        found_idx = mid;
                        break;
                    }
                    else if (mid_key < target_key)
                    {
                        left = mid + 1;
                    }
                    else
                    {
                        right = mid - 1;
                    }
                }

                voxel_edge_to_vert_idx[idx * 12 + e] = found_idx;
            }
            else
            {
                voxel_edge_to_vert_idx[idx * 12 + e] = -1; // Inactive edge
            }
        }
    }

    template <typename IndexType>
    __global__ void get_num_vd_kernel(
        const uint8_t *__restrict__ used_voxel_code,
        IndexType n_used_voxels,
        IndexType *__restrict__ num_vd_array)
    {
        IndexType idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n_used_voxels)
        {
            // Look up the exact number of dual vertices needed for this configuration
            num_vd_array[idx] = num_vd_table[used_voxel_code[idx]];
        }
    }

    template <typename Scalar, typename IndexType>
    __global__ void compute_dual_vertices_kernel(
        const uint8_t *__restrict__ used_voxel_code,
        const IndexType *__restrict__ voxel_edge_to_vert_idx,
        const IndexType *__restrict__ vd_prefix_sum,
        const Vertex<Scalar> *__restrict__ edge_zero_crossings,
        const Vertex<Scalar> *__restrict__ edge_zero_colors,
        IndexType n_used_voxels,
        Vertex<Scalar> *__restrict__ out_verts,
        Vertex<Scalar> *__restrict__ out_colors,
        bool with_colors)
    {
        IndexType idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n_used_voxels)
            return;

        uint8_t case_id = used_voxel_code[idx];
        int num_vd = num_vd_table[case_id]; // Updated table name

        if (num_vd == 0)
            return;

        IndexType out_start_idx = vd_prefix_sum[idx];

        // Loop over the dual vertices this voxel is responsible for
        for (int i = 0; i < num_vd; ++i)
        {
            Vertex<Scalar> p = {0.0f, 0.0f, 0.0f};
            Vertex<Scalar> c = {0.0f, 0.0f, 0.0f};
            int valid_edges_count = 0;

            // Up to 7 edges are used to average a single dual vertex
            for (int e = 0; e < 7; ++e)
            {
                // Updated to use the nested 3D array!
                int local_edge = dmc_table[case_id][i][e];

                if (local_edge == -1)
                    break; // -1 means no more edges for this dual vertex

                // Where is this edge in the global zero-crossing array?
                IndexType unique_edge_idx = voxel_edge_to_vert_idx[idx * 12 + local_edge];

                if (unique_edge_idx != -1)
                {
                    Vertex<Scalar> edge_p = edge_zero_crossings[unique_edge_idx];
                    p.x += edge_p.x;
                    p.y += edge_p.y;
                    p.z += edge_p.z;

                    if (with_colors)
                    {
                        Vertex<Scalar> edge_c = edge_zero_colors[unique_edge_idx];
                        c.x += edge_c.x;
                        c.y += edge_c.y;
                        c.z += edge_c.z;
                    }
                    valid_edges_count++;
                }
            }

            // Average the positions to find the centroid
            if (valid_edges_count > 0)
            {
                Scalar inv_count = Scalar(1.0) / Scalar(valid_edges_count);
                p.x = p.x * inv_count;
                p.y = p.y * inv_count;
                p.z = p.z * inv_count;

                if (with_colors)
                {
                    c.x = c.x * inv_count;
                    c.y = c.y * inv_count;
                    c.z = c.z * inv_count;
                }
            }

            out_verts[out_start_idx + i] = p;
            if (with_colors)
            {
                out_colors[out_start_idx + i] = c;
            }
        }
    }

    template <typename IndexType>
    __global__ void connect_quads_kernel(
        const uint8_t *__restrict__ used_voxel_code,
        const IndexType *__restrict__ voxel_edge_to_vert_idx,
        const IndexType *__restrict__ vd_prefix_sum,
        IndexType n_used_voxels,
        IndexType *__restrict__ quads)
    {
        IndexType idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n_used_voxels)
            return;

        uint8_t code = used_voxel_code[idx];
        IndexType vd_start = vd_prefix_sum[idx];
        
        // Fetch the number of dual vertices in this specific voxel
        int num_vd = num_vd_table[code]; 

        for (int e = 0; e < 12; ++e)
        {
            IndexType u_idx = voxel_edge_to_vert_idx[idx * 12 + e];

            if (u_idx != (IndexType)-1)
            {
                // We must search the table to find exactly WHICH of the dual 
                // vertices in this voxel owns this specific edge.
                int vd_offset = 0;
                for (int i = 0; i < num_vd; ++i) {
                    for (int k = 0; k < 7; ++k) {
                        if (dmc_table[code][i][k] == e) {
                            vd_offset = i;
                            break;
                        }
                    }
                }

                int local_v0 = c_edge_v0[e];
                int local_v1 = c_edge_v1[e];
                
                // The Global Min Rule (Prevents Slot Collisions)
                int min_local_v = min(local_v0, local_v1);
                bool min_v_inside = (code & (1 << min_local_v)) != 0;

                int x0 = local_v0 & 1;
                int y0 = (local_v0 >> 1) & 1;
                int z0 = (local_v0 >> 2) & 1;

                int x1 = local_v1 & 1;
                int y1 = (local_v1 >> 1) & 1;

                int slot = 0;

                // Dynamically determine slot
                if (x0 != x1) {
                    if (y0 == 0 && z0 == 0) slot = 0;
                    else if (y0 == 1 && z0 == 0) slot = 1;
                    else if (y0 == 1 && z0 == 1) slot = 2;
                    else slot = 3;
                } else if (y0 != y1) {
                    if (z0 == 0 && x0 == 0) slot = 0;
                    else if (z0 == 1 && x0 == 0) slot = 1;
                    else if (z0 == 1 && x0 == 1) slot = 2;
                    else slot = 3;
                } else {
                    if (x0 == 0 && y0 == 0) slot = 0;
                    else if (x0 == 1 && y0 == 0) slot = 1;
                    else if (x0 == 1 && y0 == 1) slot = 2;
                    else slot = 3;
                }

                if (!min_v_inside) {
                    slot = (4 - slot) % 4;
                }

                // Add the offset to ensure the edge attaches to the CORRECT dual vertex!
                quads[u_idx * 4 + slot] = vd_start + vd_offset;
            }
        }
    }

    template <typename Scalar, typename IndexType>
    __host__ void DMC<Scalar, IndexType>::forward(
        Vertex<Scalar> const *grid_vertices,
        Vertex<Scalar> const *grid_colors,
        IndexType const *voxels,
        Scalar const *values,
        IndexType n_voxels,
        Scalar iso,
        int device)
    {
        IndexType threads = 256;
        IndexType blocks = (n_voxels + threads - 1) / threads;
        bool with_colors = (grid_colors != nullptr);
        cudaSetDevice(device);

        // 0. Ensure we have enough storage for the voxel codes and prefix sums
        this->ensure_grid_storage_size(n_voxels);

        // 1. Run your Identification Kernel
        identify_active_voxels_kernel<<<blocks, threads>>>(voxels, values, n_voxels, iso, this->voxel_codes);
        CHECK_CUDA(cudaDeviceSynchronize());

        // 2. Compute prefix sum to prepare for compaction
        thrust::device_ptr<uint8_t> d_voxel_codes(this->voxel_codes);
        thrust::device_ptr<IndexType> d_temp_buffer(this->temp_buffer);

        // Map: Turn codes into 1s (active) and 0s (inactive) using your ops.h functor
        thrust::transform(d_voxel_codes, d_voxel_codes + n_voxels, d_temp_buffer, IsActiveOp());

        // Scan: Compute exclusive prefix sum
        thrust::exclusive_scan(d_temp_buffer, d_temp_buffer + n_voxels, d_temp_buffer);

        // Extract total number of used voxels
        uint8_t last_flag;
        IndexType last_sum;
        CHECK_CUDA(cudaMemcpy(&last_flag, this->voxel_codes + n_voxels - 1, sizeof(uint8_t), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(&last_sum, this->temp_buffer + n_voxels - 1, sizeof(IndexType), cudaMemcpyDeviceToHost));
        this->n_used_voxels = last_sum + ((last_flag > 0 && last_flag < 255) ? 1 : 0);

        if (this->n_used_voxels == 0)
        {
            this->n_verts = 0;
            this->n_quads = 0;
            return;
        }

        this->ensure_used_voxel_storage_size(this->n_used_voxels);

        compact_active_voxels_kernel<IndexType><<<blocks, threads>>>(
            this->voxel_codes, this->temp_buffer, n_voxels,
            this->used_voxel_index, this->used_voxel_code);
        CHECK_CUDA(cudaDeviceSynchronize());

        // 3. Extract active edges
        size_t raw_edge_count = this->n_used_voxels * 12;
        this->ensure_edge_storage_size(raw_edge_count, with_colors);

        int blocks_used = (this->n_used_voxels + threads - 1) / threads;

        extract_active_edges_kernel<IndexType><<<blocks_used, threads>>>(
            voxels, this->used_voxel_index, this->used_voxel_code,
            this->n_used_voxels, this->unique_edges);
        CHECK_CUDA(cudaDeviceSynchronize());

        // 4. Sort and Unique to find unique edges
        thrust::device_ptr<EdgeKey<IndexType>> d_edges(this->unique_edges);

        // 4a. Physically delete all inactive sentinels from the array (O(N))
        auto valid_end = thrust::remove_if(thrust::device, d_edges, d_edges + raw_edge_count, IsSentinelEdge<IndexType>());

        // 4b. Sort ONLY the valid edges (O(V log V))
        thrust::sort(thrust::device, d_edges, valid_end);

        // 4c. Remove duplicates
        auto unique_end = thrust::unique(thrust::device, d_edges, valid_end);

        // 4d. The exact number of valid, unique edges is the pointer distance
        this->n_quads = unique_end - d_edges;
        this->ensure_quad_storage_size(this->n_quads);

        // 5. Compute zero-crossings for unique edges
        IndexType n_unique_edges = this->n_quads;

        int blocks_edges = (n_unique_edges + threads - 1) / threads;

        if (n_unique_edges > 0)
        {
            compute_edge_zero_crossings_kernel<Scalar, IndexType><<<blocks_edges, threads>>>(
                this->unique_edges,
                grid_vertices,
                grid_colors,
                values,
                iso,
                n_unique_edges,
                this->edge_zero_crossings,
                this->edge_zero_colors,
                with_colors);
            CHECK_CUDA(cudaDeviceSynchronize());

            // 6. Map voxel edges to unique edge indices for dual vertex construction
            map_voxel_edges_to_unique_idx_kernel<IndexType><<<blocks_used, threads>>>(
                voxels,
                this->used_voxel_index,
                this->used_voxel_code,
                this->unique_edges,
                this->n_used_voxels,
                n_unique_edges,
                this->voxel_edge_to_vert_idx);
            CHECK_CUDA(cudaDeviceSynchronize());
        }

        // 7. For each active voxel, determine how many dual vertices it contributes and compute prefix sum for output indexing
        get_num_vd_kernel<IndexType><<<blocks_used, threads>>>(
            this->used_voxel_code, this->n_used_voxels, this->temp_buffer);
        CHECK_CUDA(cudaDeviceSynchronize());

        // Extract the last element BEFORE scan
        IndexType last_num_vd = 0;
        CHECK_CUDA(cudaMemcpy(&last_num_vd, this->temp_buffer + this->n_used_voxels - 1, sizeof(IndexType), cudaMemcpyDeviceToHost));

        // Exclusive Scan to get the memory offset indices
        thrust::device_ptr<IndexType> d_vd_prefix(this->temp_buffer);
        thrust::exclusive_scan(d_vd_prefix, d_vd_prefix + this->n_used_voxels, d_vd_prefix);

        // Extract total number of dual vertices
        IndexType last_scan_offset = 0;
        CHECK_CUDA(cudaMemcpy(&last_scan_offset, this->temp_buffer + this->n_used_voxels - 1, sizeof(IndexType), cudaMemcpyDeviceToHost));

        this->n_verts = last_scan_offset + last_num_vd;

        if (this->n_verts == 0)
            return;

        this->ensure_vert_storage_size(this->n_verts, with_colors);

        compute_dual_vertices_kernel<Scalar, IndexType><<<blocks_used, threads>>>(
            this->used_voxel_code,
            this->voxel_edge_to_vert_idx,
            this->temp_buffer, // Now holds the offset indices!
            this->edge_zero_crossings,
            this->edge_zero_colors,
            this->n_used_voxels,
            this->verts,
            this->out_colors,
            with_colors);
        CHECK_CUDA(cudaDeviceSynchronize());

        // 8. Connect dual vertices into quads
        connect_quads_kernel<IndexType><<<blocks_used, threads>>>(
            this->used_voxel_code,
            this->voxel_edge_to_vert_idx,
            this->temp_buffer, // vd_prefix_sum offsets
            this->n_used_voxels,
            this->quads);
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    template <typename Scalar, typename IndexType>
    __global__ void backward_dual_to_edges_kernel(
        const uint8_t *__restrict__ used_voxel_code,
        const IndexType *__restrict__ voxel_edge_to_vert_idx,
        const IndexType *__restrict__ vd_prefix_sum,
        const Vertex<Scalar> *__restrict__ adj_verts,
        const Vertex<Scalar> *__restrict__ adj_colors,
        IndexType n_used_voxels,
        Vertex<Scalar> *__restrict__ adj_edge_crossings,
        Vertex<Scalar> *__restrict__ adj_edge_colors,
        bool with_colors)
    {
        IndexType idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n_used_voxels)
            return;

        uint8_t case_id = used_voxel_code[idx];
        int num_vd = num_vd_table[case_id];

        if (num_vd == 0)
            return;

        IndexType vd_start = vd_prefix_sum[idx];

        for (int i = 0; i < num_vd; ++i)
        {
            int valid_edges_count = 0;
            for (int e = 0; e < 7; ++e)
            {
                int local_edge = dmc_table[case_id][i][e];
                if (local_edge == -1)
                    break;
                if (voxel_edge_to_vert_idx[idx * 12 + local_edge] != -1)
                {
                    valid_edges_count++;
                }
            }

            if (valid_edges_count == 0)
                continue;

            Vertex<Scalar> grad_p = adj_verts[vd_start + i];
            Vertex<Scalar> grad_c = {0, 0, 0};
            if (with_colors)
                grad_c = adj_colors[vd_start + i];

            Scalar inv_count = Scalar(1.0) / Scalar(valid_edges_count);
            grad_p *= inv_count;
            if (with_colors)
                grad_c *= inv_count;

            for (int e = 0; e < 7; ++e)
            {
                int local_edge = dmc_table[case_id][i][e];
                if (local_edge == -1)
                    break;

                IndexType u_idx = voxel_edge_to_vert_idx[idx * 12 + local_edge];
                if (u_idx != -1)
                {
                    // Atomics must remain component-wise
                    atomicAdd(&(adj_edge_crossings[u_idx].x), grad_p.x);
                    atomicAdd(&(adj_edge_crossings[u_idx].y), grad_p.y);
                    atomicAdd(&(adj_edge_crossings[u_idx].z), grad_p.z);

                    if (with_colors)
                    {
                        atomicAdd(&(adj_edge_colors[u_idx].x), grad_c.x);
                        atomicAdd(&(adj_edge_colors[u_idx].y), grad_c.y);
                        atomicAdd(&(adj_edge_colors[u_idx].z), grad_c.z);
                    }
                }
            }
        }
    }

    template <typename Scalar, typename IndexType>
    __global__ void backward_edges_to_grid_kernel(
        const EdgeKey<IndexType> *__restrict__ unique_edges,
        const Vertex<Scalar> *__restrict__ grid_vertices,
        const Scalar *__restrict__ values,
        const Vertex<Scalar> *__restrict__ adj_edge_crossings,
        const Vertex<Scalar> *__restrict__ adj_edge_colors,
        IndexType n_unique_edges,
        Scalar *__restrict__ adj_values,
        Vertex<Scalar> *__restrict__ adj_grid_colors,
        Scalar iso,
        bool with_colors)
    {
        IndexType idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n_unique_edges)
            return;

        EdgeKey<IndexType> edge = unique_edges[idx];
        IndexType v0_idx = edge.v0;
        IndexType v1_idx = edge.v1;

        Vertex<Scalar> p0 = grid_vertices[v0_idx];
        Vertex<Scalar> p1 = grid_vertices[v1_idx];
        Scalar val0 = values[v0_idx];
        Scalar val1 = values[v1_idx];

        const Scalar EPS = Scalar(1e-5);
        Scalar t;
        if (d_abs(iso - val0) < EPS)
            t = Scalar(0.0);
        else if (d_abs(iso - val1) < EPS)
            t = Scalar(1.0);
        else if (d_abs(val0 - val1) < EPS)
            t = Scalar(0.5);
        else
            t = clamp((iso - val0) / (val1 - val0), Scalar(0.0), Scalar(1.0));

        Vertex<Scalar> grad_p = adj_edge_crossings[idx];

        Scalar dot_product = grad_p.dot(p1 - p0);

        Scalar val_diff = val1 - val0;
        if (d_abs(val_diff) > EPS)
        {
            Scalar inv_diff = Scalar(1.0) / val_diff;
            Scalar grad_v0 = dot_product * ((t - Scalar(1.0)) * inv_diff);
            Scalar grad_v1 = dot_product * (-t * inv_diff);

            atomicAdd(&adj_values[v0_idx], grad_v0);
            atomicAdd(&adj_values[v1_idx], grad_v1);
        }

        if (with_colors)
        {
            Vertex<Scalar> grad_c = adj_edge_colors[idx];
            Scalar weight0 = Scalar(1.0) - t;
            Scalar weight1 = t;

            Vertex<Scalar> c0_contrib = grad_c * weight0;
            Vertex<Scalar> c1_contrib = grad_c * weight1;

            atomicAdd(&(adj_grid_colors[v0_idx].x), c0_contrib.x);
            atomicAdd(&(adj_grid_colors[v0_idx].y), c0_contrib.y);
            atomicAdd(&(adj_grid_colors[v0_idx].z), c0_contrib.z);

            atomicAdd(&(adj_grid_colors[v1_idx].x), c1_contrib.x);
            atomicAdd(&(adj_grid_colors[v1_idx].y), c1_contrib.y);
            atomicAdd(&(adj_grid_colors[v1_idx].z), c1_contrib.z);
        }
    }

    template <typename Scalar, typename IndexType>
    __host__ void DMC<Scalar, IndexType>::backward(
        Vertex<Scalar> const *grid_vertices,
        Vertex<Scalar> const *grid_colors,
        Scalar const *values,
        Vertex<Scalar> const *adj_verts,
        Vertex<Scalar> const *adj_colors,
        Scalar *adj_values,
        Vertex<Scalar> *adj_grid_colors,
        Scalar iso,
        int device)
    {
        cudaSetDevice(device);

        // Safety check: if no vertices were generated in the forward pass,
        // there are no gradients to propagate!
        if (this->n_verts == 0 || this->n_used_voxels == 0)
            return;

        // 1. Propagate gradients from Dual Vertices -> Shared Edges
        bool with_colors = (adj_colors != nullptr && adj_grid_colors != nullptr);

        IndexType n_unique_edges = this->n_quads;

        Vertex<Scalar> *d_adj_edge_crossings = nullptr;
        Vertex<Scalar> *d_adj_edge_colors = nullptr;

        CHECK_CUDA(cudaMalloc((void **)&d_adj_edge_crossings, n_unique_edges * sizeof(Vertex<Scalar>)));
        CHECK_CUDA(cudaMemset(d_adj_edge_crossings, 0, n_unique_edges * sizeof(Vertex<Scalar>)));

        if (with_colors)
        {
            CHECK_CUDA(cudaMalloc((void **)&d_adj_edge_colors, n_unique_edges * sizeof(Vertex<Scalar>)));
            CHECK_CUDA(cudaMemset(d_adj_edge_colors, 0, n_unique_edges * sizeof(Vertex<Scalar>)));
        }

        IndexType threads = 256;
        IndexType blocks_used = (this->n_used_voxels + threads - 1) / threads;

        backward_dual_to_edges_kernel<Scalar, IndexType><<<blocks_used, threads>>>(
            this->used_voxel_code,
            this->voxel_edge_to_vert_idx,
            this->temp_buffer,
            adj_verts,
            adj_colors,
            this->n_used_voxels,
            d_adj_edge_crossings,
            d_adj_edge_colors,
            with_colors);

        CHECK_CUDA(cudaDeviceSynchronize());

        IndexType blocks_edges = (n_unique_edges + threads - 1) / threads;

        backward_edges_to_grid_kernel<Scalar, IndexType><<<blocks_edges, threads>>>(
            this->unique_edges,
            grid_vertices,
            values,
            d_adj_edge_crossings,
            d_adj_edge_colors,
            n_unique_edges,
            adj_values,
            adj_grid_colors,
            iso,
            with_colors);
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaFree(d_adj_edge_crossings));
        if (with_colors)
        {
            CHECK_CUDA(cudaFree(d_adj_edge_colors));
        }
    }

    template struct DMC<float, int>;
    template struct DMC<float, long long>;
}