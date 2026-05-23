cmake_minimum_required(VERSION 3.18)
project(MarchingCubes CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

# Enable CUDA language
enable_language(CUDA)

# Set CUDA toolkit path from conda environment
set(CMAKE_CUDA_COMPILER $ENV{CONDA_PREFIX}/bin/nvcc)
set(CUDA_TOOLKIT_ROOT_DIR $ENV{CONDA_PREFIX}/targets/x86_64-linux)

# Specify CUDA architectures (adjust based on your GPU)
# 89 = RTX 4090, includes native double atomics
set(CMAKE_CUDA_ARCHITECTURES 52 61 70 75 80 86 89)

# Add the executable
add_executable(mc_forward examples/cuda/mc_forward.cu march/csrc/mc.cu)
add_executable(mc_backward examples/cuda/mc_backward.cu march/csrc/mc.cu)
add_executable(mc_backward_color examples/cuda/mc_backward_color.cu march/csrc/mc.cu)
add_executable(pc_to_voxel_grid examples/cuda/pc_to_voxel_grid.cu march/csrc/grid.cu)
add_executable(pc_to_voxel_grid_chunk examples/cuda/pc_to_voxel_grid_chunk.cu march/csrc/grid.cu)

# Link CUDA libraries
target_link_libraries(mc_forward PRIVATE cudart)
target_link_libraries(mc_backward PRIVATE cudart)
target_link_libraries(mc_backward_color PRIVATE cudart)
target_link_libraries(pc_to_voxel_grid PRIVATE cudart)
target_link_libraries(pc_to_voxel_grid_chunk PRIVATE cudart)

# Set CUDA architectures
set_target_properties(mc_forward PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(mc_backward PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(mc_backward_color PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(pc_to_voxel_grid PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(pc_to_voxel_grid_chunk PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# Include directories
target_include_directories(mc_forward PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/march/csrc ${CUDA_TOOLKIT_ROOT_DIR}/include)
target_include_directories(mc_backward PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/march/csrc ${CUDA_TOOLKIT_ROOT_DIR}/include)
target_include_directories(mc_backward_color PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/march/csrc ${CUDA_TOOLKIT_ROOT_DIR}/include)
target_include_directories(pc_to_voxel_grid PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/march/csrc ${CUDA_TOOLKIT_ROOT_DIR}/include)
target_include_directories(pc_to_voxel_grid_chunk PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/march/csrc ${CUDA_TOOLKIT_ROOT_DIR}/include)

# Output directory
set_target_properties(mc_forward PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set_target_properties(mc_backward PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set_target_properties(mc_backward_color PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set_target_properties(pc_to_voxel_grid PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set_target_properties(pc_to_voxel_grid_chunk PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
