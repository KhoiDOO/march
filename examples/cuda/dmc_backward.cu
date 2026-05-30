#include "dmc.h"
#include "primitive.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>

using primitive::Vertex;
using primitive::Triangle;
using namespace dmc;

// Generate a 3x3x3 grid of vertices (8 voxels total)
void generate_test_data(
    std::vector<Vertex<float>>& grid_vertices,
    std::vector<Vertex<float>>& grid_colors,
    std::vector<float>& values
) {
    grid_vertices.clear();
    grid_colors.clear();
    values.clear();

    // Generate 27 vertices for a 3x3x3 grid
    for (int z = 0; z <= 2; ++z) {
        for (int y = 0; y <= 2; ++y) {
            for (int x = 0; x <= 2; ++x) {
                grid_vertices.push_back({(float)x, (float)y, (float)z});
                
                // Color gradient based on spatial position
                grid_colors.push_back({x / 2.0f, y / 2.0f, z / 2.0f});
                
                // The ONLY positive value is the exact center vertex (1, 1, 1)
                if (x == 1 && y == 1 && z == 1) {
                    values.push_back(1.0f);  // Inside the surface
                } else {
                    values.push_back(-1.0f); // Outside the surface
                }
            }
        }
    }
}

// Generate the 8 voxels that connect the 3x3x3 grid
void generate_voxels(std::vector<int>& voxel_indices) {
    voxel_indices.clear();
    
    // Create a 2x2x2 arrangement of voxels
    for (int z = 0; z < 2; ++z) {
        for (int y = 0; y < 2; ++y) {
            for (int x = 0; x < 2; ++x) {
                // Calculate the flat indices for the 8 corners of this voxel
                int v0 = (x)     + (y) * 3     + (z) * 9;
                int v1 = (x + 1) + (y) * 3     + (z) * 9;
                int v2 = (x)     + (y + 1) * 3 + (z) * 9;
                int v3 = (x + 1) + (y + 1) * 3 + (z) * 9;
                int v4 = (x)     + (y) * 3     + (z + 1) * 9;
                int v5 = (x + 1) + (y) * 3     + (z + 1) * 9;
                int v6 = (x)     + (y + 1) * 3 + (z + 1) * 9;
                int v7 = (x + 1) + (y + 1) * 3 + (z + 1) * 9;

                voxel_indices.push_back(v0);
                voxel_indices.push_back(v1);
                voxel_indices.push_back(v2);
                voxel_indices.push_back(v3);
                voxel_indices.push_back(v4);
                voxel_indices.push_back(v5);
                voxel_indices.push_back(v6);
                voxel_indices.push_back(v7);
            }
        }
    }
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

    DMC<float, int> dmc;
    std::cout << "Running forward pass without color data..." << std::endl;
    dmc.forward(d_grid_vertices, nullptr, d_voxels, d_values, n_voxels, iso_value, device);
    
    std::cout << "Running forward pass with color data..." << std::endl;
    dmc.forward(d_grid_vertices, d_grid_colors, d_voxels, d_values, n_voxels, iso_value, device);
    
    std::cout << "Forward pass completed!" << std::endl;
    std::cout << "  Active voxels: " << dmc.n_used_voxels << std::endl;
    std::cout << "  Generated vertices: " << dmc.n_verts << std::endl;
    std::cout << "  Generated quads: " << dmc.n_quads << std::endl;

    int total_indices = dmc.n_quads * 4;
    
    // Copy results back to host
    std::vector<Vertex<float>> result_verts(dmc.n_verts);
    std::vector<int> result_quads(total_indices);
    
    std::cout << "Copying results back to host..." << std::endl;
    cudaMemcpy(result_verts.data(), dmc.verts, dmc.n_verts * sizeof(Vertex<float>), cudaMemcpyDeviceToHost);
    cudaMemcpy(result_quads.data(), dmc.quads, total_indices * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Print results
    std::cout << "\nResult Vertices (first 10):" << std::endl;
    for (int i = 0; i < std::min(10, (int)dmc.n_verts); ++i) {
        std::cout << "  V" << i << ": (" << result_verts[i].x << ", " 
                  << result_verts[i].y << ", " << result_verts[i].z << ")" << std::endl;
    }
    
    std::cout << "\nResult Quads (first 10):" << std::endl;
    for (int i = 0; i < std::min(40, (int)total_indices); i += 4) {
        std::cout << "  Q" << i/4 << ": (" 
            << result_quads[i] << ", " 
            << result_quads[i+1] << ", " 
            << result_quads[i+2] << ", " 
            << result_quads[i+3] << ")" 
            << std::endl;
    }
    
    // Save to OBJ file
    // std::cout << "\nSaving results to dmc_forward.obj..." << std::endl;
    // std::ofstream obj_file("dmc_forward.obj");
    // for (const auto& v : result_verts) {
    //     obj_file << "v " << v.x << " " << v.y << " " << v.z << "\n";
    // }
    // for (int i = 0; i < total_indices; i += 4) {
    //     obj_file << "f " << (result_quads[i]+1) << " " 
    //              << (result_quads[i+1]+1) << " " 
    //              << (result_quads[i+2]+1) << " " 
    //              << (result_quads[i+3]+1) << "\n";
    // }
    // obj_file.close();

    // backward pass
    std::cout << "\n=== Marching Cubes Backward Pass ===" << std::endl;

    std::vector<Vertex<float>> adj_verts(dmc.n_verts);
    for (int i = 0; i < dmc.n_verts; ++i) {
        // Example: gradient is proportional to vertex position
        adj_verts[i].x = result_verts[i].x;
        adj_verts[i].y = result_verts[i].y;
        adj_verts[i].z = result_verts[i].z;
    }

    Vertex<float>* d_adj_verts;
    Vertex<float>* d_adj_colors;
    Vertex<float>* d_adj_grid_colors;
    float* d_adj_values;

    std::cout << "Allocating device memory for gradients..." << std::endl;
    // 1. Dual Vertex Gradients are sized to dmc.n_verts (8)
    cudaMalloc(&d_adj_verts, dmc.n_verts * sizeof(Vertex<float>));
    cudaMalloc(&d_adj_colors, dmc.n_verts * sizeof(Vertex<float>));

    // 2. Grid Gradients are sized to n_vertices (27)
    cudaMalloc(&d_adj_grid_colors, n_vertices * sizeof(Vertex<float>));
    cudaMalloc(&d_adj_values, n_vertices * sizeof(float));

    std::cout << "Copying incoming neural net gradients to device..." << std::endl;
    cudaMemcpy(d_adj_verts, adj_verts.data(), dmc.n_verts * sizeof(Vertex<float>), cudaMemcpyHostToDevice);
    // Using adj_verts data as dummy incoming color gradients for the test
    cudaMemcpy(d_adj_colors, adj_verts.data(), dmc.n_verts * sizeof(Vertex<float>), cudaMemcpyHostToDevice); 

    std::cout << "Zeroing out grid gradient accumulators..." << std::endl;
    // CRITICAL: Must be initialized to 0.0 for atomicAdd to work correctly!
    cudaMemset(d_adj_grid_colors, 0, n_vertices * sizeof(Vertex<float>));
    cudaMemset(d_adj_values, 0, n_vertices * sizeof(float));

    std::cout << "Running backward pass..." << std::endl;

    dmc.backward(
        d_grid_vertices, 
        d_grid_colors, 
        d_values, 
        d_adj_verts, 
        d_adj_colors, 
        d_adj_values, 
        d_adj_grid_colors, 
        iso_value, 
        device
    );

    std::cout << "Backward pass completed!" << std::endl;

    // Copy gradients back to host
    std::vector<float> result_adj_values(n_vertices);
    std::vector<Vertex<float>> result_adj_grid_colors(n_vertices);
    
    std::cout << "Copying adjoint values back to host..." << std::endl;
    cudaMemcpy(result_adj_values.data(), d_adj_values, n_vertices * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(result_adj_grid_colors.data(), d_adj_grid_colors, n_vertices * sizeof(Vertex<float>), cudaMemcpyDeviceToHost);

    // Print backward results
    std::cout << "\nBackward Result Gradients (adj_values) for all vertices:" << std::endl;
    for (int i = 0; i < n_vertices; ++i) {
        std::cout << "  dL/dvalue[" << i << "] = " << result_adj_values[i] << std::endl;
    }

    std::cout << "\nBackward Result Gradients (adj_grid_colors) for all vertices:" << std::endl;
    for (int i = 0; i < n_vertices; ++i) {
        std::cout << "  dL/dcolor[" << i << "] = (" << result_adj_grid_colors[i].x << ", " << result_adj_grid_colors[i].y << ", " << result_adj_grid_colors[i].z << ")" << std::endl;
    }

    // Save gradients
    std::cout << "Saving backward results to mc_backward_gradients.txt..." << std::endl;
    std::ofstream grad_file("mc_backward_gradients.txt");
    grad_file << "# Adjoint values (gradients w.r.t. input scalar values)\n";
    grad_file << "# Index, Gradient\n";
    for (int i = 0; i < n_vertices; ++i) {
        grad_file << i << ", " << result_adj_values[i] << "\n";
    }
    grad_file << "# Adjoint colors (gradients w.r.t. input vertex colors)\n";
    grad_file << "# Index, Gradient\n";
    for (int i = 0; i < n_vertices; ++i) {
        grad_file << i << ", " << result_adj_grid_colors[i].x << ", " << result_adj_grid_colors[i].y << ", " << result_adj_grid_colors[i].z << "\n";
    }
    grad_file.close();
    
    // Cleanup
    std::cout << "Cleaning up device memory..." << std::endl;
    cudaFree(d_grid_vertices);
    cudaFree(d_grid_colors);
    cudaFree(d_values);
    cudaFree(d_voxels);
    cudaFree(d_adj_verts);
    cudaFree(d_adj_colors);
    cudaFree(d_adj_values);
    cudaFree(d_adj_grid_colors);
    std::cout << "\n=== Done ===" << std::endl;
    return 0;
}
