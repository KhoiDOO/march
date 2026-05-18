#include "mc.h"

#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cuda_fp16.h>

namespace py = pybind11;

// Helper macros to check tensor properties
#define CHECK_CUDA(x) AT_ASSERTM(x.options().device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Helper functions to convert between PyTorch tensors and raw pointers
template <typename T>
T* get_tensor_ptr(torch::Tensor& tensor) { return tensor.data_ptr<T>(); }

template <typename T>
const T* get_tensor_ptr_const(const torch::Tensor& tensor) { return tensor.data_ptr<T>(); }

namespace mc_wrapper {
    template <typename Scalar, typename IndexType>
    class MC_Wrapper {
        mc::MC<Scalar, IndexType> mc;

        static_assert(std::is_same<Scalar, float>() || std::is_same<Scalar, __half>());
        static_assert(std::is_same<IndexType, long>() || std::is_same<IndexType, int>());

    public:
        // ~MC() {
        //     // Destructor will automatically call the destructor of mc, which frees GPU memory
        //     cudaDeviceSynchronize();
        //     cudaFree(mc.temp_buffer);
        //     cudaFree(mc.cube_codes);
        //     cudaFree(mc.used_cube_code);
        //     cudaFree(mc.used_cube_index);
        //     cudaFree(mc.cube_edge_to_vert_idx);
        //     cudaFree(mc.used_to_first_mc_tri);
        //     cudaFree(mc.unique_edges);
        //     cudaFree(mc.verts);
        //     cudaFree(mc.tris);
        // }

        std::tuple<torch::Tensor, torch::Tensor> forward(
            torch::Tensor grid_vertices, // N * 3 array of grid vertex positions
            torch::Tensor cubes, // (N-1) * 8 array of cube vertex indices
            torch::Tensor values, // N array of scalar values at grid vertices
            Scalar iso
        ) {
            CHECK_INPUT(grid_vertices);
            CHECK_INPUT(cubes);
            CHECK_INPUT(values);

            int n_cubes = cubes.size(0);
            int device = grid_vertices.device().index();

            torch::ScalarType scalarType;   
            if constexpr (std::is_same<Scalar, float>())
            {
                scalarType = torch::kFloat;
            }
            else
            {
                scalarType = torch::kHalf;
            }
            TORCH_INTERNAL_ASSERT(grid_vertices.dtype() == scalarType, "grid type must match the mc class");
            TORCH_INTERNAL_ASSERT(values.dtype() == scalarType, "values type must match the mc class");

            torch::ScalarType indexType = torch::kInt;
            if constexpr (std::is_same<IndexType, int>())
            {
                indexType = torch::kInt;
            }
            else
            {
                indexType = torch::kLong;
            }
            TORCH_INTERNAL_ASSERT(cubes.dtype() == indexType, "cubes type must match the mc class");

            mc.forward(
                reinterpret_cast<mc::Vertex<Scalar> const *>(get_tensor_ptr_const<Scalar>(grid_vertices)),
                get_tensor_ptr_const<IndexType>(cubes),
                get_tensor_ptr_const<Scalar>(values),
                n_cubes,
                iso,
                device // device ID
            );

            int n_verts = mc.n_verts;
            int n_tris = mc.n_tris / 3;

            auto options_float = torch::TensorOptions().dtype(scalarType).device(grid_vertices.device());
            auto options_int = torch::TensorOptions().dtype(indexType).device(cubes.device());

            torch::Tensor verts_tensor = torch::from_blob(
                mc.verts, {n_verts, 3}, options_float
            ).clone();

            torch::Tensor tris_tensor = torch::from_blob(
                mc.tris, {n_tris, 3}, options_int
            ).clone();

            return std::make_tuple(verts_tensor, tris_tensor);
        }

        void backward(
            torch::Tensor grid_vertices,
            torch::Tensor values,
            torch::Tensor adj_verts,
            torch::Tensor adj_values,
            Scalar iso
        ) {
            // Make tensors contiguous if needed
            grid_vertices = grid_vertices.contiguous();
            values = values.contiguous();
            adj_verts = adj_verts.contiguous();
            adj_values = adj_values.contiguous();
            
            CHECK_INPUT(grid_vertices);
            CHECK_INPUT(values);
            CHECK_INPUT(adj_verts);
            CHECK_INPUT(adj_values);

            int device = grid_vertices.device().index();

            torch::ScalarType scalarType;
            if constexpr (std::is_same<Scalar, float>())
            {
                scalarType = torch::kFloat;
            }
            else
            {
                scalarType = torch::kHalf;
            }
            TORCH_INTERNAL_ASSERT(grid_vertices.dtype() == scalarType, "grid type must match the mc class");
            TORCH_INTERNAL_ASSERT(values.dtype() == scalarType, "values type must match the mc class");
            TORCH_INTERNAL_ASSERT(adj_verts.dtype() == scalarType, "adj_verts type must match the mc class");
            TORCH_INTERNAL_ASSERT(adj_values.dtype() == scalarType, "adj_values type must match the mc class");
            
            if (mc.n_verts == 0) {
                // No vertices were generated, so we can skip the backward pass
                return;
            }

            mc.backward(
                reinterpret_cast<mc::Vertex<Scalar> const *>(get_tensor_ptr_const<Scalar>(grid_vertices)),
                get_tensor_ptr_const<Scalar>(values),
                reinterpret_cast<mc::Vertex<Scalar> const *>(get_tensor_ptr_const<Scalar>(adj_verts)),
                get_tensor_ptr<Scalar>(adj_values),
                iso,
                device // device ID
            );
        }
    };
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    pybind11::class_<mc_wrapper::MC_Wrapper<float, int>>(m, "MCF")
        .def(pybind11::init<>())
        .def("forward", pybind11::overload_cast<torch::Tensor, torch::Tensor, torch::Tensor, float>(&mc_wrapper::MC_Wrapper<float, int>::forward))
        .def("backward", pybind11::overload_cast<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, float>(&mc_wrapper::MC_Wrapper<float, int>::backward));
    // pybind11::class_<mc_wrapper::MC_Wrapper<__half, int>>(m, "MCH")
    //     .def(pybind11::init<>())
    //     .def("forward", pybind11::overload_cast<torch::Tensor, torch::Tensor, torch::Tensor, __half>(&mc_wrapper::MC_Wrapper<__half, int>::forward))
    //     .def("backward", pybind11::overload_cast<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, __half>(&mc_wrapper::MC_Wrapper<__half, int>::backward));
}