#include "mc.h"
#include "primitive.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>

using primitive::Vertex;
using primitive::Triangle;
using namespace mc;

// Generate test data: 4 cubes with specific vertices and values
void generate_test_data(
    std::vector<Vertex<float>>& grid_vertices,
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

// Generate cube connectivity for the 4-cube test case
void generate_cubes(
    std::vector<int>& cube_indices
) {
    // 4 cubes, each with 8 vertex indices
    cube_indices = {
        // Cube 0
        0, 1, 2, 3, 4, 5, 6, 7,
        // Cube 1
        1, 8, 3, 9, 5, 10, 7, 11,
        // Cube 2
        12, 13, 14, 15, 0, 1, 2, 3,
        // Cube 3
        13, 16, 15, 17, 1, 8, 3, 9
    };
}

int main() {
    std::cout << "=== Marching Cubes Backward Pass ===" << std::endl;
    
    // Parameters
    float iso_value = 0.0f;  // Isosurface value
    int device = 0;
    
    // Host data
    std::vector<Vertex<float>> grid_vertices;
    std::vector<float> values;
    std::vector<int> cube_indices;
    
    std::cout << "Generating test data (4 cubes example)..." << std::endl;
    generate_test_data(grid_vertices, values);
    generate_cubes(cube_indices);
    
    int n_vertices = grid_vertices.size();
    int n_cubes = cube_indices.size() / 8;
    
    std::cout << "  Grid vertices: " << n_vertices << std::endl;
    std::cout << "  Number of cubes: " << n_cubes << std::endl;
    std::cout << "  Isosurface value: " << iso_value << std::endl;
    
    // Device data
    Vertex<float>* d_grid_vertices;
    float* d_values;
    int* d_cubes;
    
    cudaSetDevice(device);
    
    std::cout << "\nAllocating device memory..." << std::endl;
    cudaMalloc(&d_grid_vertices, n_vertices * sizeof(Vertex<float>));
    cudaMalloc(&d_values, n_vertices * sizeof(float));
    cudaMalloc(&d_cubes, cube_indices.size() * sizeof(int));
    
    std::cout << "Copying data to device..." << std::endl;
    cudaMemcpy(d_grid_vertices, grid_vertices.data(), n_vertices * sizeof(Vertex<float>), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, values.data(), n_vertices * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cubes, cube_indices.data(), cube_indices.size() * sizeof(int), cudaMemcpyHostToDevice);
    
    // ========== FORWARD PASS ==========
    std::cout << "\n=== Forward Pass ===" << std::endl;
    std::cout << "Running forward pass..." << std::endl;
    MC<float, int> mc;
    mc.forward(d_grid_vertices, d_cubes, d_values, n_cubes, iso_value, device);
    
    std::cout << "Forward pass completed!" << std::endl;
    std::cout << "  Active cubes: " << mc.n_used_cubes << std::endl;
    std::cout << "  Generated vertices: " << mc.n_verts << std::endl;
    std::cout << "  Generated triangles: " << mc.n_tris / 3 << std::endl;
    
    // Copy forward results back to host
    std::vector<Vertex<float>> result_verts(mc.n_verts);
    std::vector<int> result_tris(mc.n_tris);
    
    std::cout << "Copying forward results back to host..." << std::endl;
    cudaMemcpy(result_verts.data(), mc.verts, mc.n_verts * sizeof(Vertex<float>), cudaMemcpyDeviceToHost);
    cudaMemcpy(result_tris.data(), mc.tris, mc.n_tris * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Print forward results
    std::cout << "\nForward Result Vertices (first 10):" << std::endl;
    for (int i = 0; i < std::min(10, (int)mc.n_verts); ++i) {
        std::cout << "  V" << i << ": (" << result_verts[i].x << ", " 
                  << result_verts[i].y << ", " << result_verts[i].z << ")" << std::endl;
    }
    
    std::cout << "\nForward Result Triangles (first 10):" << std::endl;
    for (int i = 0; i < std::min(30, (int)mc.n_tris); i += 3) {
        std::cout << "  T" << i/3 << ": (" << result_tris[i] << ", " 
                  << result_tris[i+1] << ", " << result_tris[i+2] << ")" << std::endl;
    }
    
    // ========== BACKWARD PASS ==========
    std::cout << "\n=== Backward Pass ===" << std::endl;
    
    // Create gradient data (adjoint of vertices)
    // In this test, we use simple gradients (e.g., all ones, or based on vertex position)
    std::vector<Vertex<float>> adj_verts(mc.n_verts);
    for (int i = 0; i < mc.n_verts; ++i) {
        // Example: gradient is proportional to vertex position
        adj_verts[i].x = result_verts[i].x;
        adj_verts[i].y = result_verts[i].y;
        adj_verts[i].z = result_verts[i].z;
    }
    
    // Allocate device memory for gradients
    Vertex<float>* d_adj_verts;
    float* d_adj_values;
    
    std::cout << "Allocating device memory for gradients..." << std::endl;
    cudaMalloc(&d_adj_verts, mc.n_verts * sizeof(Vertex<float>));
    cudaMalloc(&d_adj_values, n_vertices * sizeof(float));
    
    std::cout << "Copying adjoint vertices to device..." << std::endl;
    cudaMemcpy(d_adj_verts, adj_verts.data(), mc.n_verts * sizeof(Vertex<float>), cudaMemcpyHostToDevice);
    
    std::cout << "Running backward pass..." << std::endl;
    mc.backward(d_grid_vertices, d_values, d_adj_verts, d_adj_values, iso_value, device);
    
    std::cout << "Backward pass completed!" << std::endl;
    
    // Copy gradients back to host
    std::vector<float> result_adj_values(n_vertices);
    
    std::cout << "Copying adjoint values back to host..." << std::endl;
    cudaMemcpy(result_adj_values.data(), d_adj_values, n_vertices * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Print backward results
    std::cout << "\nBackward Result Gradients (adj_values) for all vertices:" << std::endl;
    for (int i = 0; i < n_vertices; ++i) {
        std::cout << "  dL/dvalue[" << i << "] = " << result_adj_values[i] << std::endl;
    }
    
    // Save results to file
    std::cout << "\nSaving forward results to mc_forward.obj..." << std::endl;
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
    
    // Save gradients
    std::cout << "Saving backward results to mc_backward_gradients.txt..." << std::endl;
    std::ofstream grad_file("mc_backward_gradients.txt");
    grad_file << "# Adjoint values (gradients w.r.t. input scalar values)\n";
    grad_file << "# Index, Gradient\n";
    for (int i = 0; i < n_vertices; ++i) {
        grad_file << i << ", " << result_adj_values[i] << "\n";
    }
    grad_file.close();
    
    // Cleanup
    std::cout << "Cleaning up device memory..." << std::endl;
    cudaFree(d_grid_vertices);
    cudaFree(d_values);
    cudaFree(d_cubes);
    cudaFree(d_adj_verts);
    cudaFree(d_adj_values);
    
    std::cout << "\n=== Done ===" << std::endl;
    return 0;
}
