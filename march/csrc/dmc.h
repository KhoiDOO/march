#ifndef DMC_H
#define DMC_H

#include "primitive.h"
#include <cstdint>
#include <cuda_runtime.h>

using primitive::EdgeKey;
using primitive::Quad;
using primitive::Vertex;

namespace dmc
{

    template <typename Scalar, typename IndexType>
    struct DMC
    {

        IndexType n_used_voxels{0};
        IndexType n_verts{0};
        IndexType n_quads{0};

        // Temp storage
        size_t allocated_voxel_count{};
        IndexType *__restrict__ temp_buffer{};

        // voxel
        uint8_t *__restrict__ voxel_codes{}; // voxel code for each cell

        // used voxel
        size_t allocated_used_voxel_count{};
        uint8_t *__restrict__ used_voxel_code{};          // used voxel to voxel code
        IndexType *__restrict__ used_voxel_index{};       // used voxel to voxel index
        IndexType *__restrict__ voxel_edge_to_vert_idx{}; // voxel to unique edge index

        // unique edge to vert
        EdgeKey<IndexType> *__restrict__ unique_edges{}; // [n_verts] array storing the pair of grid vertex indices
        Vertex<Scalar> *__restrict__ edge_zero_crossings{}; // [n_unique_edges] array storing the zero-crossing positions
        Vertex<Scalar> *__restrict__ edge_zero_colors{}; // [n_unique_edges] array storing the colors at zero-crossing positions

        // output
        size_t allocated_vert_count{};
        size_t allocated_quad_count{};
        size_t allocated_edge_count{};
        Vertex<Scalar> *__restrict__ verts{};      // output verts
        Vertex<Scalar> *__restrict__ out_colors{}; // output vert colors
        IndexType *__restrict__ quads{};           // output quads

        __host__ void ensure_grid_storage_size(size_t n_voxels);
        __host__ void ensure_used_voxel_storage_size(size_t n_used_voxels);
        __host__ void ensure_edge_storage_size(size_t n_edges, bool with_colors);
        __host__ void ensure_vert_storage_size(size_t n_dual_verts, bool with_colors);
        __host__ void ensure_quad_storage_size(size_t n_quads);

        __host__ void forward(
            Vertex<Scalar> const *grid_vertices, // N * 3 array of grid vertex positions
            Vertex<Scalar> const *grid_colors,   // N * 3 array of grid vertex colors
            IndexType const *voxels,             // (N-1) * 8 array of voxel vertex indices
            Scalar const *values,                // N array of scalar values at grid vertices
            IndexType n_voxels,
            Scalar iso,
            int device);

        __host__ void backward(
            Vertex<Scalar> const *grid_vertices,
            Vertex<Scalar> const *grid_colors,
            Scalar const *values,
            Vertex<Scalar> const *adj_verts,
            Vertex<Scalar> const *adj_colors,
            Scalar *adj_values,
            Vertex<Scalar> *adj_grid_colors,
            Scalar iso,
            int device);

        __host__ ~DMC()
        {
            if (temp_buffer)
                cudaFree(temp_buffer);
            if (voxel_codes)
                cudaFree(voxel_codes);
            if (used_voxel_code)
                cudaFree(used_voxel_code);
            if (used_voxel_index)
                cudaFree(used_voxel_index);
            if (voxel_edge_to_vert_idx)
                cudaFree(voxel_edge_to_vert_idx);
            if (unique_edges)
                cudaFree(unique_edges);
            if (edge_zero_crossings)
                cudaFree(edge_zero_crossings);
            if (edge_zero_colors)
                cudaFree(edge_zero_colors);
            if (verts)
                cudaFree(verts);
            if (quads)
                cudaFree(quads);
            if (out_colors)
                cudaFree(out_colors);
        }
    };
}

#endif // DMC_H