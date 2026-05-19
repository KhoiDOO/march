#include "primitive.h"
#include <cstdint>
#include <cuda_runtime.h>

using primitive::Vertex;
using primitive::Triangle;


//  Coordinate system
//
//       z
//       |
//       |
//       |
//       0-----x
//      /
//     /
//    y
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

namespace mc {

    template <typename Scalar, typename IndexType>
    struct MC {

        IndexType n_used_cubes{0};
        IndexType n_verts{0};
        IndexType n_tris{0};

        // Temp storage
        size_t allocated_cube_count{};
        IndexType *__restrict__ temp_buffer{};

        // cube
        uint8_t *__restrict__ cube_codes{};           // cube code for each cell

        // used cube
        size_t allocated_used_cube_count{};
        uint8_t *__restrict__ used_cube_code{};           // used cube to cube code
        IndexType *__restrict__ used_cube_index{};        // used cube to cube index
        IndexType *__restrict__ cube_edge_to_vert_idx{};    // cube to unique edge index
        IndexType *__restrict__ used_to_first_mc_tri{};   // used cube to mc tri index

        // unique edge to vert
        long long *__restrict__ unique_edges{}; // [n_verts] array storing the pair of grid vertex indices

        // output
        size_t allocated_vert_count{};
        size_t allocated_tri_count{};
        Vertex<Scalar> *__restrict__ verts{}; // output verts
        IndexType *__restrict__ tris{}; // output triangles

        __host__ void ensure_grid_storage_size(size_t n_cubes);
        __host__ void ensure_used_cube_storage_size(size_t n_used_cubes);
        __host__ void ensure_vert_storage_size(size_t n_verts);
        __host__ void ensure_tri_storage_size(size_t n_tris);

        __host__ void forward(
            Vertex<Scalar> const *grid_vertices, // N * 3 array of grid vertex positions
            IndexType const *cubes, // (N-1) * 8 array of cube vertex indices
            Scalar const *values, // N array of scalar values at grid vertices
            IndexType n_cubes, 
            Scalar iso, 
            int device
        );

        __host__ void backward(
            Vertex<Scalar> const *grid_vertices,
            Scalar const *values,
            Vertex<Scalar> const *adj_verts,
            Scalar *adj_values,
            Scalar iso,
            int device
        );

        __host__ ~MC() {
            if (temp_buffer) cudaFree(temp_buffer);
            if (cube_codes) cudaFree(cube_codes);
            if (used_cube_code) cudaFree(used_cube_code);
            if (used_cube_index) cudaFree(used_cube_index);
            if (cube_edge_to_vert_idx) cudaFree(cube_edge_to_vert_idx);
            if (used_to_first_mc_tri) cudaFree(used_to_first_mc_tri);
            if (unique_edges) cudaFree(unique_edges);
            if (verts) cudaFree(verts);
            if (tris) cudaFree(tris);
        }
    };
}