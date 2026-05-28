#ifndef MC_H
#define MC_H

#include "primitive.h"
#include <cstdint>
#include <cuda_runtime.h>

using primitive::EdgeKey;
using primitive::Triangle;
using primitive::Vertex;

//  Coordinate system
//
//       y
//       |
//       |
//       |
//       0-----x
//      /
//     /
//    z
//

// Cell Corners
// (Corners are voxels. Number correspond to Morton codes of corner coordinates)
//
//       4-------------------5
//      /|                  /|
//     / |                 / |
//    /  |                /  |
//   6-------------------7   |
//   |   |               |   |
//   |   |               |   |
//   |   |               |   |
//   |   |               |   |
//   |   0---------------|---1
//   |  /                |  /
//   | /                 | /
//   |/                  |/
//   2-------------------3
//

//         Cell Edges
//
//       o--------4----------o
//      /|                  /|
//     7 |                 5 |
//    /  |                /  |
//   o--------6----------o   |
//   |   8               |   9
//   |   |               |   |
//   |   |               |   |
//   11  |               10  |
//   |   o--------0------|---o
//   |  /                |  /
//   | 3                 | 1
//   |/                  |/
//   o--------2----------o
//

namespace mc
{

    template <typename Scalar, typename IndexType>
    struct MC
    {

        IndexType n_used_voxels{0};
        IndexType n_verts{0};
        IndexType n_tris{0};

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
        IndexType *__restrict__ used_to_first_mc_tri{};   // used voxel to mc tri index

        // unique edge to vert
        EdgeKey<IndexType> *__restrict__ unique_edges{}; // [n_verts] array storing the pair of grid vertex indices

        // output
        size_t allocated_vert_count{};
        size_t allocated_tri_count{};
        Vertex<Scalar> *__restrict__ verts{};      // output verts
        Vertex<Scalar> *__restrict__ out_colors{}; // output vert colors
        IndexType *__restrict__ tris{};            // output triangles

        __host__ void ensure_grid_storage_size(size_t n_voxels);
        __host__ void ensure_used_voxel_storage_size(size_t n_used_voxels);
        __host__ void ensure_vert_storage_size(size_t n_verts, bool with_colors);
        __host__ void ensure_tri_storage_size(size_t n_tris);

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

        __host__ ~MC()
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
            if (used_to_first_mc_tri)
                cudaFree(used_to_first_mc_tri);
            if (unique_edges)
                cudaFree(unique_edges);
            if (verts)
                cudaFree(verts);
            if (tris)
                cudaFree(tris);
            if (out_colors)
                cudaFree(out_colors);
        }
    };
}

#endif // MC_H