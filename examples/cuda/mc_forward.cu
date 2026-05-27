#include "mc.h"
#include "primitive.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>

using primitive::Vertex;
using primitive::Triangle;
using namespace mc;

// Generate test data: 4 voxels with specific vertices and values
void generate_test_data(
    std::vector<Vertex<float>>& grid_vertices,
    std::vector<Vertex<float>>& grid_colors,
    std::vector<float>& values
) {
    // Grid vertices (18 vertices)
    grid_vertices = {
        {0, 0, 0},      // v0
        {1, 0, 0},      // v1
        {0, 1, 0},      // v2
        {1, 1, 0},      // v3
        {0, 0, 1},      // v4
        {1, 0, 1},      // v5
        {0, 1, 1},      // v6
        {1, 1, 1},      // v7
        {2, 0, 0},      // v8
        {2, 1, 0},      // v9
        {2, 0, 1},      // v10
        {2, 1, 1},      // v11
        {0, 0, -1},     // v12
        {1, 0, -1},     // v13
        {0, 1, -1},     // v14
        {1, 1, -1},     // v15
        {2, 0, -1},     // v16
        {2, 1, -1}      // v17
    };

    // Grid colors (18 vertices)
    grid_colors = {
        {1, 0, 0},      // v0
        {0, 1, 0},      // v1
        {0, 0, 1},      // v2
        {1, 1, 0},      // v3
        {1, 0, 1},      // v4
        {0, 1, 1},      // v5
        {1, 1, 1},      // v6
        {0, 0, 0},      // v7
        {1, 0.5f, 0},   // v8
        {0.5f, 1, 0},   // v9
        {0.5f, 0, 1},   // v10
        {0, 0.5f, 1},   // v11
        {1, 0, 0.5f},   // v12
        {0, 1, 0.5f},   // v13
        {0.5f, 1, 0},   // v14
        {0.5f, 0.5f, 0.5f}, // v15
        {0.5f, 0, 0},   // v16
        {0, 0.5f, 0}    // v17
    };
    
    // Scalar values (18 values)
    values = {
        1,          // v0
        -0.5f,      // v1
        1,          // v2
        -0.5f,      // v3
        1,          // v4
        1,          // v5
        1,          // v6
        1,          // v7
        1,          // v8
        1,          // v9
        1,          // v10
        1,          // v11
        1,          // v12
        -0.5f,      // v13
        1,          // v14
        -0.5f,      // v15
        1,          // v16
        1           // v17
    };
}

// Generate voxel connectivity for the 4-voxel test case
void generate_voxels(
    std::vector<int>& voxel_indices
) {
    // 4 voxels, each with 8 vertex indices
    voxel_indices = {
        // Voxel 0
        0, 1, 2, 3, 4, 5, 6, 7,
        // Voxel 1
        1, 8, 3, 9, 5, 10, 7, 11,
        // Voxel 2
        12, 13, 14, 15, 0, 1, 2, 3,
        // Voxel 3
        13, 16, 15, 17, 1, 8, 3, 9
    };
}

int main() {
    std::cout << "=== Marching Cubes Forward Pass ===" << std::endl;
    
    // Parameters
    float iso_value = 0.0f;  // Isosurface value
    int device = 0;
    
    // Host data
    std::vector<Vertex<float>> grid_vertices;
    std::vector<Vertex<float>> grid_colors;
    std::vector<float> values;
    std::vector<int> voxel_indices;
    
    std::cout << "Generating test data (4 voxels example)..." << std::endl;
    generate_test_data(grid_vertices, grid_colors, values);
    generate_voxels(voxel_indices);
    
    int n_vertices = grid_vertices.size();
    int n_voxels = voxel_indices.size() / 8;
    
    std::cout << "  Grid vertices: " << n_vertices << std::endl;
    std::cout << "  Number of voxels: " << n_voxels << std::endl;
    std::cout << "  Isosurface value: " << iso_value << std::endl;
    
    // Device data
    Vertex<float>* d_grid_vertices;
    Vertex<float>* d_grid_colors;
    float* d_values;
    int* d_voxels;
    
    cudaSetDevice(device);
    
    std::cout << "Allocating device memory..." << std::endl;
    cudaMalloc(&d_grid_vertices, n_vertices * sizeof(Vertex<float>));
    cudaMalloc(&d_grid_colors, n_vertices * sizeof(Vertex<float>));
    cudaMalloc(&d_values, n_vertices * sizeof(float));
    cudaMalloc(&d_voxels, voxel_indices.size() * sizeof(int));
    
    std::cout << "Copying data to device..." << std::endl;
    cudaMemcpy(d_grid_vertices, grid_vertices.data(), n_vertices * sizeof(Vertex<float>), cudaMemcpyHostToDevice);
    cudaMemcpy(d_grid_colors, grid_colors.data(), n_vertices * sizeof(Vertex<float>), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, values.data(), n_vertices * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_voxels, voxel_indices.data(), voxel_indices.size() * sizeof(int), cudaMemcpyHostToDevice);

    MC<float, int> mc;
    std::cout << "Running forward pass without color data..." << std::endl;
    mc.forward(d_grid_vertices, nullptr, d_voxels, d_values, n_voxels, iso_value, device);
    
    std::cout << "Running forward pass with color data..." << std::endl;
    mc.forward(d_grid_vertices, d_grid_colors, d_voxels, d_values, n_voxels, iso_value, device);
    
    std::cout << "Forward pass completed!" << std::endl;
    std::cout << "  Active voxels: " << mc.n_used_voxels << std::endl;
    std::cout << "  Generated vertices: " << mc.n_verts << std::endl;
    std::cout << "  Generated triangles: " << mc.n_tris / 3 << std::endl;
    
    // Copy results back to host
    std::vector<Vertex<float>> result_verts(mc.n_verts);
    std::vector<int> result_tris(mc.n_tris);
    
    std::cout << "Copying results back to host..." << std::endl;
    cudaMemcpy(result_verts.data(), mc.verts, mc.n_verts * sizeof(Vertex<float>), cudaMemcpyDeviceToHost);
    cudaMemcpy(result_tris.data(), mc.tris, mc.n_tris * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Print results
    std::cout << "\nResult Vertices (first 10):" << std::endl;
    for (int i = 0; i < std::min(10, (int)mc.n_verts); ++i) {
        std::cout << "  V" << i << ": (" << result_verts[i].x << ", " 
                  << result_verts[i].y << ", " << result_verts[i].z << ")" << std::endl;
    }
    
    std::cout << "\nResult Triangles (first 10):" << std::endl;
    for (int i = 0; i < std::min(30, (int)mc.n_tris); i += 3) {
        std::cout << "  T" << i/3 << ": (" << result_tris[i] << ", " 
                  << result_tris[i+1] << ", " << result_tris[i+2] << ")" << std::endl;
    }
    
    // Save to OBJ file
    std::cout << "\nSaving results to mc_forward.obj..." << std::endl;
    std::ofstream obj_file("mc_forward.obj");
    for (const auto& v : result_verts) {
        obj_file << "v " << v.x << " " << v.y << " " << v.z << "\n";
    }
    for (int i = 0; i < mc.n_tris; i += 3) {
        obj_file << "f " << (result_tris[i]+1) << " " 
                 << (result_tris[i+1]+1) << " " 
                 << (result_tris[i+2]+1) << "\n";
    }
    obj_file.close();


    
    // Cleanup
    std::cout << "Cleaning up device memory..." << std::endl;
    cudaFree(d_grid_vertices);
    cudaFree(d_grid_colors);
    cudaFree(d_values);
    cudaFree(d_voxels);
    
    std::cout << "\n=== Done ===" << std::endl;
    return 0;
}
