#include "grid.h"
#include "primitive.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <fstream>

using primitive::Vertex;
using namespace grid;

// Generate a simple point cloud that covers a small block of voxels
void generate_test_points(std::vector<Vertex<float>>& points) {
    points = {
        {0.1f, 0.1f, 0.1f},
        {0.9f, 0.1f, 0.1f},
        {0.1f, 0.9f, 0.1f},
        {0.9f, 0.9f, 0.1f},
        {0.1f, 0.1f, 0.9f},
        {0.9f, 0.1f, 0.9f},
        {0.1f, 0.9f, 0.9f},
        {0.9f, 0.9f, 0.9f},
        // some additional points to populate neighbouring voxels
        {1.5f, 0.5f, 0.5f},
        {2.5f, 1.5f, 0.5f},
        {0.5f, 2.5f, 1.5f}
    };
}

int main() {
    std::cout << "=== pc_to_voxel_grid CUDA example ===\n";

    int device = 0;
    cudaSetDevice(device);

    // Parameters for voxelization
    int res_x = 4, res_y = 4, res_z = 4;
    int k_threshold = 1;
    int num_keep = 0;
    float r = 1.0f; // bounding-box truncation / scaling factor

    // Prepare host points
    std::vector<Vertex<float>> h_points;
    generate_test_points(h_points);
    int n_points = (int)h_points.size();

    std::cout << "Num input points: " << n_points << "\n";

    // Copy points to device
    Vertex<float>* d_points = nullptr;
    cudaMalloc(&d_points, n_points * sizeof(Vertex<float>));
    cudaMemcpy(d_points, h_points.data(), n_points * sizeof(Vertex<float>), cudaMemcpyHostToDevice);

    // Outputs (device pointers will be allocated inside the function)
    Vertex<float>* d_out_vertices = nullptr;
    int out_num_vertices = 0;
    int* d_out_cubes = nullptr;
    int out_num_cubes = 0;

    // Call the voxelization function (explicit instantiation exists for float,int)
    grid::pc_to_voxel_grid<float,int>(
        d_points, n_points,
        res_x, res_y, res_z,
        k_threshold, num_keep, r,
        &d_out_vertices, &out_num_vertices,
        &d_out_cubes, &out_num_cubes,
        device
    );

    std::cout << "Output vertices: " << out_num_vertices << "\n";
    std::cout << "Output cubes: " << out_num_cubes << "\n";

    // Copy results back to host if any
    std::vector<Vertex<float>> h_out_vertices;
    std::vector<int> h_out_cubes;
    if (out_num_vertices > 0) {
        h_out_vertices.resize(out_num_vertices);
        cudaMemcpy(h_out_vertices.data(), d_out_vertices, out_num_vertices * sizeof(Vertex<float>), cudaMemcpyDeviceToHost);
    }
    if (out_num_cubes > 0) {
        h_out_cubes.resize(out_num_cubes * 8);
        cudaMemcpy(h_out_cubes.data(), d_out_cubes, out_num_cubes * 8 * sizeof(int), cudaMemcpyDeviceToHost);
    }

    // Save vertices to OBJ for quick inspection
    std::ofstream obj("pc_to_voxel_grid_output.obj");
    for (const auto &v : h_out_vertices) {
        obj << "v " << v.x << " " << v.y << " " << v.z << "\n";
    }
    // Save cube indices as faces (as 8-tuples per cube, not standard OBJ faces)
    obj << "# cubes (8 vertex indices per cube):\n";
    for (int i = 0; i < (int)h_out_cubes.size(); i += 8) {
        obj << "# cube ";
        for (int j = 0; j < 8; ++j) obj << h_out_cubes[i + j] << (j+1==8?"":" ");
        obj << "\n";
    }
    obj.close();

    std::cout << "Wrote pc_to_voxel_grid_output.obj\n";

    // Cleanup
    cudaFree(d_points);
    if (d_out_vertices) cudaFree(d_out_vertices);
    if (d_out_cubes) cudaFree(d_out_cubes);

    std::cout << "=== Done ===\n";
    return 0;
}
