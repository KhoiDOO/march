#include "primitive.h"
#include <cstdint>
#include <cuda_runtime.h>

using primitive::Vertex;

namespace grid {

    template <typename Scalar, typename IndexType>
    void pc_to_voxel_grid(
        Vertex<Scalar> const *points,
        IndexType num_points,
        IndexType res_x, 
        IndexType res_y, 
        IndexType res_z, 
        int k_threshold,
        int num_keep,
        float rmi_x, float rmi_y, float rmi_z, float rma_x, float rma_y, float rma_z,
        Vertex<Scalar>** out_vertices,
        IndexType* out_num_vertices,
        IndexType** out_cubes,
        IndexType* out_num_cubes,
        int device
    );

    template <typename Scalar, typename IndexType>
    void pc_to_voxel_grid_chunk(
        Vertex<Scalar> const *points,
        IndexType num_points,
        IndexType res_x, 
        IndexType res_y, 
        IndexType res_z,
        int chunk_size, 
        int k_threshold,
        int num_keep,
        float rmi_x, float rmi_y, float rmi_z, float rma_x, float rma_y, float rma_z,
        Vertex<Scalar>** out_vertices,
        IndexType* out_num_vertices,
        IndexType** out_cubes,
        IndexType* out_num_cubes,
        int device
    );
}