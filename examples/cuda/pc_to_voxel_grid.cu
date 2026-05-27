#include "grid.h"
#include "primitive.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <fstream>

using primitive::Vertex;
using namespace grid;

// Generate a random point cloud inside [0, span) for each axis
void generate_test_points(std::vector<Vertex<float>>& points, int n_points, float span=1024.0f, unsigned int seed=42u) {
    points.clear();
    points.reserve(n_points);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0.0f, span);
    for (int i = 0; i < n_points; ++i) {
        points.push_back({dist(rng), dist(rng), dist(rng)});
    }
}

int main() {
    std::cout << "=== pc_to_voxel_grid CUDA example ===\n";

    int device = 0;
    cudaSetDevice(device);

    // Parameters for voxelization (large test)
    int res_x = 128, res_y = 128, res_z = 128;
    int k_threshold = 1;
    int num_keep = 0;
    float rmi_x = 0.2f, rmi_y = 0.0f, rmi_z = 1.0f; // min corner of the bounding box
    float rma_x = 1.0f, rma_y = 1.0f, rma_z = 1.0f; // max corner of the bounding box

    // Prepare host points (100k random samples over the 0..1024 cube)
    const int n_points = 100000;
    std::vector<Vertex<float>> h_points;
    generate_test_points(h_points, n_points, 1024.0f, 42u);

    std::cout << "Num input points: " << n_points << "\n";

    // Copy points to device
    Vertex<float>* d_points = nullptr;
    cudaMalloc(&d_points, n_points * sizeof(Vertex<float>));
    cudaMemcpy(d_points, h_points.data(), n_points * sizeof(Vertex<float>), cudaMemcpyHostToDevice);

    std::cout << "Copied points to device\n";

    // Outputs (device pointers will be allocated inside the function)
    Vertex<float>* d_out_vertices = nullptr;
    int out_num_vertices = 0;
    int* d_out_voxels = nullptr;
    int out_num_voxels = 0;

    // Call the voxelization function (explicit instantiation exists for float, int)

    std::cout << "Calling pc_to_voxel_grid (explicit instantiation for float, int)\n";

    grid::pc_to_voxel_grid<float,int>(
        d_points, n_points,
        res_x, res_y, res_z,
        k_threshold, num_keep, rmi_x, rmi_y, rmi_z, rma_x, rma_y, rma_z,
        &d_out_vertices, &out_num_vertices,
        &d_out_voxels, &out_num_voxels,
        device
    );

    std::cout << "Output vertices: " << out_num_vertices << "\n";
    std::cout << "Output voxels: " << out_num_voxels << "\n";

    // Copy results back to host if any
    std::vector<Vertex<float>> h_out_vertices;
    std::vector<int> h_out_voxels;
    if (out_num_vertices > 0) {
        h_out_vertices.resize(out_num_vertices);
        cudaMemcpy(h_out_vertices.data(), d_out_vertices, out_num_vertices * sizeof(Vertex<float>), cudaMemcpyDeviceToHost);
    }
    if (out_num_voxels > 0) {
        h_out_voxels.resize(out_num_voxels * 8);
        cudaMemcpy(h_out_voxels.data(), d_out_voxels, out_num_voxels * 8 * sizeof(int), cudaMemcpyDeviceToHost);
    }

    // Save vertices to OBJ for quick inspection
    std::ofstream obj("pc_to_voxel_grid_output.obj");
    for (const auto &v : h_out_vertices) {
        obj << "v " << v.x << " " << v.y << " " << v.z << "\n";
    }
    // Save voxel indices as faces (as 8-tuples per voxel, not standard OBJ faces)
    obj << "# voxels (8 vertex indices per voxel):\n";
    for (int i = 0; i < (int)h_out_voxels.size(); i += 8) {
        obj << "# voxel ";
        for (int j = 0; j < 8; ++j) obj << h_out_voxels[i + j] << (j+1==8?"":" ");
        obj << "\n";
    }
    obj.close();

    std::cout << "Wrote pc_to_voxel_grid_output.obj\n";

    // Call the voxelization function (explicit instantiation exists for float, long long)

    std::cout << "Calling pc_to_voxel_grid (explicit instantiation for float, long long)\n";

    Vertex<float>* d_out_vertices_ll = nullptr;
    long long out_num_vertices_ll = 0;
    long long* d_out_voxels_ll = nullptr;
    long long out_num_voxels_ll = 0;

    grid::pc_to_voxel_grid<float,long long>(
        d_points, n_points,
        res_x, res_y, res_z,
        k_threshold, num_keep, rmi_x, rmi_y, rmi_z, rma_x, rma_y, rma_z,
        &d_out_vertices_ll, &out_num_vertices_ll,
        &d_out_voxels_ll, &out_num_voxels_ll,
        device
    );

    std::cout << "Output vertices (long long): " << out_num_vertices_ll << "\n";
    std::cout << "Output voxels (long long): " << out_num_voxels_ll << "\n";

    // Cleanup
    cudaFree(d_points);
    if (d_out_vertices) cudaFree(d_out_vertices);
    if (d_out_voxels) cudaFree(d_out_voxels);
    if (d_out_vertices_ll) cudaFree(d_out_vertices_ll);
    if (d_out_voxels_ll) cudaFree(d_out_voxels_ll);

    std::cout << "=== Done ===\n";
    return 0;
}
