#include "mc.h"
#include "primitive.h"
#include "constants/mc_tables_y.h"
#include "ops.h"
#include "kernels/grid.h"

#include <cstdint>
#include <cstdio>

#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/unique.h>
#include <thrust/binary_search.h>
#include <thrust/iterator/transform_iterator.h>

namespace mc {

    template <typename IndexType>
    __global__ void extract_active_edges_kernel(
        const IndexType *voxels,
        const IndexType *used_voxel_index,
        const uint8_t* used_voxel_code,
        EdgeKey<IndexType>* active_edges,
        IndexType n_used_voxels
    ) {
        IndexType active_voxel_idx = (IndexType) blockIdx.x * blockDim.x + threadIdx.x;
        if (active_voxel_idx >= n_used_voxels) return;

        IndexType voxel_idx = used_voxel_index[active_voxel_idx];
        uint8_t code = used_voxel_code[active_voxel_idx];
        const IndexType *v_ptr = &voxels[voxel_idx * 8]; // 8 vertex indices of the voxel

        int face_start = firstMarchingCubesId[code];
        int face_num = firstMarchingCubesId[code + 1] - face_start;

        for (int i = 0; i < 12; i++) {
            active_edges[active_voxel_idx * 12 + i] = {(IndexType)-1, (IndexType)-1};
        }

        for (int i = 0; i < face_num; ++i) {

            int local_edge_id = marchingCubesIds[face_start + i];

            IndexType v0 = v_ptr[edge2vertices[local_edge_id][0]];
            IndexType v1 = v_ptr[edge2vertices[local_edge_id][1]];
            active_edges[active_voxel_idx * 12 + local_edge_id] = {min(v0, v1), max(v0, v1)};
        }
    };

    template <typename IndexType>
    __global__ void build_edge_map_kernel(
        const IndexType* voxels,
        const IndexType* used_voxel_index,
        const EdgeKey<IndexType>* unique_edges,
        IndexType* voxel_edge_to_vert_idx,
        IndexType n_used_voxels,
        IndexType n_verts
    ) {
        IndexType active_idx = (IndexType) blockIdx.x * blockDim.x + threadIdx.x;
        if (active_idx >= n_used_voxels) return;

        IndexType global_voxel_idx = used_voxel_index[active_idx];
        const IndexType *v_ptr = &voxels[global_voxel_idx * 8];

        for (int i = 0; i < 12; ++i) {
            IndexType v0 = v_ptr[edge2vertices[i][0]];
            IndexType v1 = v_ptr[edge2vertices[i][1]];
            EdgeKey<IndexType> edge_key = {min(v0, v1), max(v0, v1)};

            // Binary search in unique_edges to find the vertex index
            IndexType left = 0;
            IndexType right = n_verts - 1;
            IndexType unique_id = -1;

            while (left <= right) {
                IndexType mid = left + (right - left) / 2;
                if (unique_edges[mid] == edge_key) {
                    unique_id = mid;
                    break;
                } else if (unique_edges[mid] < edge_key) {
                    left = mid + 1;
                } else {
                    right = mid - 1;
                }
            }

            voxel_edge_to_vert_idx[active_idx * 12 + i] = unique_id;
        }
    };

    template <typename Scalar, typename IndexType>
    __global__ void interpolate_vertices_kernel(
        const EdgeKey<IndexType>* unique_edges,
        const Vertex<Scalar>* grid_vertices,
        const Vertex<Scalar>* grid_colors,
        const Scalar* values,
        IndexType n_verts,
        Scalar iso,
        Vertex<Scalar>* out_verts,
        Vertex<Scalar>* out_colors,
        bool with_colors
    ) {
        IndexType v_idx = (IndexType) blockIdx.x * blockDim.x + threadIdx.x;
        if (v_idx >= n_verts) return;

        // 1. Decode the 64-bit edge
        EdgeKey<IndexType> edge_sig = unique_edges[v_idx];
        IndexType v0_idx = edge_sig.v0;
        IndexType v1_idx = edge_sig.v1;

        // 2. Fetch positions and values
        Vertex<Scalar> p0 = grid_vertices[v0_idx];
        Vertex<Scalar> p1 = grid_vertices[v1_idx];
        Scalar val0 = values[v0_idx];
        Scalar val1 = values[v1_idx];

        Vertex<Scalar> c0, c1;
        if (with_colors) {
            c0 = grid_colors[v0_idx];
            c1 = grid_colors[v1_idx];
        }

        // 3. Interpolate (Differentiable formula)
        const Scalar EPS = Scalar(1e-5);

        Vertex<Scalar> p;
        Vertex<Scalar> c;

        if (d_abs(iso - val0) < EPS) {
            p = p0; 
            if (with_colors) c = c0;
        } 
        else if (d_abs(iso - val1) < EPS) {
            p = p1; 
            if (with_colors) c = c1;
        } 
        else if (d_abs(val0 - val1) < EPS) {
            p = p0; 
            if (with_colors) c = c0;
        }
        else {
            Scalar t = (val1 != val0) ? clamp((iso - val0) / (val1 - val0), Scalar(0.0), Scalar(1.0)) : Scalar(0.5);
            p = p0 + (p1 - p0) * t;
            if (with_colors) c = c0 + (c1 - c0) * t;
        }
            
        out_verts[v_idx] = p;
        if (with_colors) out_colors[v_idx] = c;
    };

    template <typename IndexType>
    __global__ void assemble_triangles_kernel(
        const uint8_t* used_voxel_code,
        const IndexType* tri_prefix_sum,
        const IndexType* voxel_edge_to_vert_idx,
        IndexType n_used_voxels,
        IndexType* tris
    ) {
        IndexType active_idx = (IndexType) blockIdx.x * blockDim.x + threadIdx.x;
        if (active_idx >= n_used_voxels) return;

        // 1. Get the code for this voxel
        uint8_t code = used_voxel_code[active_idx];

        // 2. Find the start/count in the FACE_TABLE
        int face_start = firstMarchingCubesId[code];
        int face_num = firstMarchingCubesId[code + 1] - face_start;

        // 3. Find where we start writing in the global 'tris' array
        int out_start = tri_prefix_sum[active_idx];

        // 4. Copy the vertex IDs using the map
        for (int i = 0; i < face_num; i++) {
            int local_edge_id = marchingCubesIds[face_start + i];
            // Use the map to get the shared unique vertex ID
            IndexType unique_v_id = voxel_edge_to_vert_idx[active_idx * 12 + local_edge_id];
            tris[out_start + i] = unique_v_id;
        }
    };

    template <typename Scalar, typename IndexType>
    void MC<Scalar, IndexType>::ensure_grid_storage_size(size_t n_voxels) {
        if (n_voxels > this->allocated_voxel_count) {
            this->allocated_voxel_count = n_voxels + n_voxels / 5; // Add 20% buffer to avoid frequent reallocations

            // Free old memory
            if (this->temp_buffer) CHECK_CUDA(cudaFree(this->temp_buffer));
            if (this->voxel_codes) CHECK_CUDA(cudaFree(this->voxel_codes));

            // Allocate new memory
            CHECK_CUDA(cudaMalloc((void **)&this->temp_buffer, this->allocated_voxel_count * sizeof(IndexType)));
            CHECK_CUDA(cudaMalloc((void **)&this->voxel_codes, this->allocated_voxel_count * sizeof(uint8_t)));
        }
    };

    template <typename Scalar, typename IndexType>
    void MC<Scalar, IndexType>::ensure_used_voxel_storage_size(size_t n_used_voxels) {
        if (n_used_voxels > this->allocated_used_voxel_count) {

            this->allocated_used_voxel_count = n_used_voxels + n_used_voxels / 5; // Add 20% buffer to avoid frequent reallocations

            // Free old memory
            if (this->used_voxel_code) CHECK_CUDA(cudaFree(this->used_voxel_code));
            if (this->used_voxel_index) CHECK_CUDA(cudaFree(this->used_voxel_index));
            if (this->used_to_first_mc_tri) CHECK_CUDA(cudaFree(this->used_to_first_mc_tri));
            if (this->voxel_edge_to_vert_idx) CHECK_CUDA(cudaFree(this->voxel_edge_to_vert_idx));

            // Allocate new memory
            CHECK_CUDA(cudaMalloc((void **)&this->used_voxel_code, this->allocated_used_voxel_count * sizeof(uint8_t)));
            CHECK_CUDA(cudaMalloc((void **)&this->used_voxel_index, this->allocated_used_voxel_count * sizeof(IndexType)));
            CHECK_CUDA(cudaMalloc((void **)&this->used_to_first_mc_tri, this->allocated_used_voxel_count * sizeof(IndexType)));
            CHECK_CUDA(cudaMalloc((void **)&this->voxel_edge_to_vert_idx, this->allocated_used_voxel_count * 12 * sizeof(IndexType)));
        }
    };

    template <typename Scalar, typename IndexType>
    void MC<Scalar, IndexType>::ensure_vert_storage_size(size_t n_verts, bool with_colors) {

        bool missing_color_buffer = (with_colors && this->out_colors == nullptr);

        if (n_verts > this->allocated_vert_count || missing_color_buffer) {
            
            if (n_verts > this->allocated_vert_count) {
                this->allocated_vert_count = n_verts + n_verts / 5;
            }  // Add 20% buffer to avoid frequent reallocations

            // Free old memory
            if (this->unique_edges) CHECK_CUDA(cudaFree(this->unique_edges));
            if (this->verts) CHECK_CUDA(cudaFree(this->verts));
            if (this->out_colors && with_colors) CHECK_CUDA(cudaFree(this->out_colors));

            // Allocate new memory
            CHECK_CUDA(cudaMalloc((void **)&this->unique_edges, this->allocated_vert_count * sizeof(EdgeKey<IndexType>)));
            CHECK_CUDA(cudaMalloc((void **)&this->verts, this->allocated_vert_count * sizeof(Vertex<Scalar>)));
            if (with_colors) {
                CHECK_CUDA(cudaMalloc((void **)&this->out_colors, this->allocated_vert_count * sizeof(Vertex<Scalar>)));
            } else {
                this->out_colors = nullptr;
            }
        }
    };

    template <typename Scalar, typename IndexType>
    void MC<Scalar, IndexType>::ensure_tri_storage_size(size_t n_tris) {
        if (n_tris > this->allocated_tri_count) {
            this->allocated_tri_count = n_tris + n_tris / 5;  // Add 20% buffer to avoid frequent reallocations

            // Free old memory
            if (this->tris) CHECK_CUDA(cudaFree(this->tris));

            // Allocate new memory
            CHECK_CUDA(cudaMalloc((void **)&this->tris, this->allocated_tri_count * sizeof(IndexType)));
        }
    };

    template <typename Scalar, typename IndexType>
    void MC<Scalar, IndexType>::forward(
        Vertex<Scalar> const *grid_vertices,
        Vertex<Scalar> const *grid_colors,
        IndexType const *voxels,
        Scalar const *values,
        IndexType n_voxels, 
        Scalar iso, 
        int device
    ) {

        IndexType threads = 256;
        IndexType blocks = (n_voxels + threads - 1) / threads;
        bool with_colors = (grid_colors != nullptr);
        cudaSetDevice(device);
        
        // 0. Ensure we have enough storage for the voxel codes and prefix sums
        this->ensure_grid_storage_size(n_voxels);

        // 1. Run your Identification Kernel
        identify_active_voxels_kernel<<<blocks, threads>>>(voxels, values, n_voxels, iso, this->voxel_codes);
        CHECK_CUDA(cudaDeviceSynchronize());

        // 2. Wrap the raw 'voxel_codes' pointer in a Thrust pointer
        thrust::device_ptr<uint8_t> d_codes(this->voxel_codes);

        // 3. Create the "Virtual" iterator
        auto active_flag_iter = thrust::make_transform_iterator(d_codes, IsActiveOp());

        // 4. Prefix Sum Output
        thrust::device_ptr<IndexType> d_prefix_sum(this->temp_buffer);

        // 5. Run Exclusive Scan (Prefix Sum) on the "Virtual" iterator
        thrust::exclusive_scan(active_flag_iter, active_flag_iter + n_voxels, d_prefix_sum);

        // 6. Get the total number of active voxels from the last element of the prefix sum + last voxel's active flag
        uint8_t last_flag;
        IndexType last_sum;
        CHECK_CUDA(cudaMemcpy(&last_flag, this->voxel_codes + n_voxels - 1, sizeof(uint8_t), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(&last_sum, this->temp_buffer + n_voxels - 1, sizeof(IndexType), cudaMemcpyDeviceToHost));
        this->n_used_voxels = last_sum + ((last_flag > 0 && last_flag < 255) ? 1 : 0);

        if (this->n_used_voxels == 0) {
            this->n_verts = 0;
            this->n_tris = 0;
            return;
        }

        this->ensure_used_voxel_storage_size(this->n_used_voxels);
        
        // 7. Run your Compaction Kernel to fill 'used_voxel_index' with the indices of active voxels
        compact_active_voxels_kernel<<<blocks, threads>>>(
            this->voxel_codes, this->temp_buffer, n_voxels, this->used_voxel_index, this->used_voxel_code);
        CHECK_CUDA(cudaDeviceSynchronize());

        // 8. Run your Extraction Kernel to fill 'active_edges' with the edges of active voxels
        EdgeKey<IndexType> *d_all_edges;
        CHECK_CUDA(cudaMalloc(&d_all_edges, this->n_used_voxels * 12 * sizeof(EdgeKey<IndexType>)));

        // 9. Run the kernel to extract active edges
        IndexType active_blocks = (this->n_used_voxels + threads - 1) / threads;
        extract_active_edges_kernel<<<active_blocks, threads>>>(
            voxels, 
            this->used_voxel_index, 
            this->used_voxel_code,
            d_all_edges, 
            this->n_used_voxels
        );
        CHECK_CUDA(cudaDeviceSynchronize());

        // 10. Sort and Unique
        thrust::device_ptr<EdgeKey<IndexType>> dev_all_edges(d_all_edges);
        thrust::sort(dev_all_edges, dev_all_edges + (this->n_used_voxels * 12));

        EdgeKey<IndexType> empty_edge = {(IndexType)-1, (IndexType)-1};
        auto valid_start = thrust::upper_bound(dev_all_edges, dev_all_edges + (this->n_used_voxels * 12), empty_edge);

        this->n_verts = thrust::distance(valid_start, thrust::unique(valid_start, dev_all_edges + (this->n_used_voxels * 12)));
        this->ensure_vert_storage_size(this->n_verts, with_colors);

        // 11. Extract the unique edges to a separate array for interpolation
        CHECK_CUDA(cudaMemcpy(this->unique_edges, valid_start.get(), this->n_verts * sizeof(EdgeKey<IndexType>), cudaMemcpyDeviceToDevice));
        CHECK_CUDA(cudaFree(d_all_edges));

        // 12. Build Edge Map
        build_edge_map_kernel<<<active_blocks, threads>>>(
            voxels, 
            this->used_voxel_index, 
            this->unique_edges, 
            this->voxel_edge_to_vert_idx, 
            this->n_used_voxels, 
            this->n_verts
        );
        CHECK_CUDA(cudaDeviceSynchronize());

        // 13. Interpolate Vertices
        IndexType vert_blocks = (this->n_verts + threads - 1) / threads;
        interpolate_vertices_kernel<<<vert_blocks, threads>>>(
            this->unique_edges,
            grid_vertices,
            grid_colors,
            values,
            this->n_verts,
            iso,
            this->verts,
            this->out_colors,
            with_colors
        );
        CHECK_CUDA(cudaDeviceSynchronize());

        // 14. Count the number of triangles for each active voxel using the FACE_TABLE
        thrust::device_ptr<uint8_t> d_used_codes(this->used_voxel_code);
        auto tri_count_iter = thrust::make_transform_iterator(d_used_codes, TriCountOp());

        thrust::device_ptr<IndexType> d_tri_prefix_sum(this->used_to_first_mc_tri);
        thrust::exclusive_scan(tri_count_iter, tri_count_iter + this->n_used_voxels, d_tri_prefix_sum);

        // 15. Get total number of triangles
        IndexType last_offset;
        uint8_t last_code;
        CHECK_CUDA(cudaMemcpy(&last_code, this->used_voxel_code + this->n_used_voxels - 1, sizeof(uint8_t), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(&last_offset, this->used_to_first_mc_tri + this->n_used_voxels - 1, sizeof(IndexType), cudaMemcpyDeviceToHost));

        // int mc_values[2];
        // CHECK_CUDA(cudaMemcpyFromSymbol(mc_values, firstMarchingCubesId, 2 * sizeof(int), (last_code) * sizeof(int)));
        // int last_tri_len = mc_values[1] - mc_values[0];
        int last_tri_len =  h_firstMarchingCubesId[last_code + 1] - h_firstMarchingCubesId[last_code];
        this->n_tris = last_offset + last_tri_len; // Total indices (divide by 3 for triangle count)
        this->ensure_tri_storage_size(this->n_tris);

        // 16. Allocate output triangle array
        assemble_triangles_kernel<<<active_blocks, threads>>>(
            this->used_voxel_code,
            this->used_to_first_mc_tri,
            this->voxel_edge_to_vert_idx,
            this->n_used_voxels,
            this->tris
        );
        CHECK_CUDA(cudaDeviceSynchronize());
    };

    template <typename Scalar, typename IndexType>
    __global__ void backward_dmc_kernel(
        const EdgeKey<IndexType>* unique_edges,
        const Scalar* grid_values,
        const Vertex<Scalar>* grid_coords,
        const Vertex<Scalar>* grid_colors,
        const Vertex<Scalar>* adj_verts,
        const Vertex<Scalar>* adj_colors,
        IndexType n_verts,
        Scalar iso,
        Scalar* adj_values,
        Vertex<Scalar>* adj_grid_colors,
        bool with_colors
    ) {
        IndexType v_idx = (IndexType) blockIdx.x * blockDim.x + threadIdx.x;
        if (v_idx >= n_verts) return;

        EdgeKey<IndexType> edge_sig = unique_edges[v_idx];
        IndexType v0_idx = edge_sig.v0;
        IndexType v1_idx = edge_sig.v1;

        Scalar v0_val = grid_values[v0_idx];
        Scalar v1_val = grid_values[v1_idx];
        Vertex<Scalar> p0 = grid_coords[v0_idx];
        Vertex<Scalar> p1 = grid_coords[v1_idx];

        Vertex<Scalar> grad_p_out = adj_verts[v_idx];

        Vertex<Scalar> c0, c1, grad_c_out;
        if (with_colors) {
            c0 = grid_colors[v0_idx];
            c1 = grid_colors[v1_idx];
            grad_c_out = adj_colors[v_idx];
        }

        Scalar diff = v1_val - v0_val;

        if (with_colors) {
            const Scalar EPS = Scalar(1e-5);
            Scalar t = Scalar(0.5);
            if (d_abs(iso - v0_val) < EPS) t = Scalar(0.0);
            else if (d_abs(iso - v1_val) < EPS) t = Scalar(1.0);
            else if (d_abs(diff) >= EPS) {
                t = clamp((iso - v0_val) / diff, Scalar(0.0), Scalar(1.0));
            }

            constexpr int CHANNELS = sizeof(Vertex<Scalar>) / sizeof(Scalar);
            Scalar t0 = Scalar(1.0) - t;
            Scalar t1 = t;
            Scalar* adj_c0_ptr = (Scalar*)&adj_grid_colors[v0_idx];
            Scalar* adj_c1_ptr = (Scalar*)&adj_grid_colors[v1_idx];
            const Scalar* grad_c_out_ptr = (const Scalar*)&grad_c_out;
            
            for (int c = 0; c < CHANNELS; ++c) {
                atomicAdd(&adj_c0_ptr[c], grad_c_out_ptr[c] * t0);
                atomicAdd(&adj_c1_ptr[c], grad_c_out_ptr[c] * t1);
            }
        }
        
        if (diff * diff < Scalar(1e-14)) return;
        Scalar dot_prod = (p1 - p0).dot(grad_p_out);
        Scalar common = dot_prod / (diff * diff);
        Scalar grad_v0 = common * (iso - v1_val);
        Scalar grad_v1 = common * (v0_val - iso);

        // Distribute the gradients back to the grid
        // Multiple unique edges share the same grid vertex, so we MUST use atomicAdd
        atomicAdd(&adj_values[v0_idx], grad_v0);
        atomicAdd(&adj_values[v1_idx], grad_v1);
    };

    template <typename Scalar, typename IndexType>
    void MC<Scalar, IndexType>::backward(
        Vertex<Scalar> const *grid_vertices,
        Vertex<Scalar> const *grid_colors,
        Scalar const *values,
        Vertex<Scalar> const *adj_verts,
        Vertex<Scalar> const *adj_colors,
        Scalar *adj_values,
        Vertex<Scalar> *adj_grid_colors,
        Scalar iso,
        int device
    ) {
        cudaSetDevice(device);
        bool with_colors = (grid_colors != nullptr && adj_colors != nullptr && adj_grid_colors != nullptr);
        
        // If no vertices were generated, there are no gradients to propagate
        if (this->n_verts == 0) return;
        IndexType threads = 256;
        IndexType blocks = (this->n_verts + threads - 1) / threads;
        backward_dmc_kernel<<<blocks, threads>>>(
            this->unique_edges,
            values,
            grid_vertices,
            grid_colors,
            adj_verts,
            adj_colors,
            this->n_verts,
            iso,
            adj_values,
            adj_grid_colors,
            with_colors
        );

        // Ensure the GPU finishes before returning to the framework
        CHECK_CUDA(cudaDeviceSynchronize());
    };

    template struct MC<float, int>;
    template struct MC<float, long long>;

    // Explicit template instantiation for kernel functions

    template __global__ void extract_active_edges_kernel<int>(
        const int*, const int*, const uint8_t*, EdgeKey<int>*, int);
    template __global__ void extract_active_edges_kernel<long long>(
        const long long*, const long long*, const uint8_t*, EdgeKey<long long>*, long long);

    template __global__ void build_edge_map_kernel<int>(
        const int*, const int*, const EdgeKey<int>*, int*, int, int);
    template __global__ void build_edge_map_kernel<long long>(
        const long long*, const long long*, const EdgeKey<long long>*, long long*, long long, long long);

    template __global__ void interpolate_vertices_kernel<float, int>(
        const EdgeKey<int>*, 
        const Vertex<float>*, 
        const Vertex<float>*, 
        const float*, 
        int, 
        float, 
        Vertex<float>*, 
        Vertex<float>*, 
        bool
    );
    template __global__ void interpolate_vertices_kernel<float, long long>(
        const EdgeKey<long long>*, 
        const Vertex<float>*, 
        const Vertex<float>*, 
        const float*, 
        long long, 
        float, 
        Vertex<float>*, 
        Vertex<float>*, 
        bool
    );

    template __global__ void assemble_triangles_kernel<int>(
        const uint8_t*, const int*, const int*, int, int*);
    template __global__ void assemble_triangles_kernel<long long>(
        const uint8_t*, const long long*, const long long*, long long, long long*);

    template __global__ void backward_dmc_kernel<float, int>(
        const EdgeKey<int>*, 
        const float*, 
        const Vertex<float>*, 
        const Vertex<float>*, 
        const Vertex<float>*, 
        const Vertex<float>*, 
        int, 
        float, 
        float*,
        Vertex<float>*,
        bool
    );
    template __global__ void backward_dmc_kernel<float, long long>(
        const EdgeKey<long long>*, 
        const float*, 
        const Vertex<float>*, 
        const Vertex<float>*, 
        const Vertex<float>*, 
        const Vertex<float>*, 
        long long, 
        float, 
        float*,
        Vertex<float>*,
        bool
    );
}

template struct primitive::Vertex<float>;
template struct primitive::Triangle<int>;
template struct primitive::EdgeKey<int>;
template struct primitive::EdgeKey<long long>;