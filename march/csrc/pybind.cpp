#include "mc.h"
#include "grid.h"
#include "primitive.h"

#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cuda_fp16.h>
#include <optional>

namespace py = pybind11;

// Helper macros to check tensor properties
#define CHECK_CUDA(x) AT_ASSERTM((x).options().device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM((x).is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Helper functions to convert between PyTorch tensors and raw pointers
template <typename T>
T* get_tensor_ptr(torch::Tensor& tensor) { 
    if constexpr (std::is_same<T, int>()) {
        return reinterpret_cast<T*>(tensor.data_ptr<int32_t>());
    } else if constexpr (std::is_same<T, long>() || std::is_same<T, long long>()) {
        return reinterpret_cast<T*>(tensor.data_ptr<int64_t>());
    } else {
        return tensor.data_ptr<T>(); 
    }
}

template <typename T>
const T* get_tensor_ptr_const(const torch::Tensor& tensor) { 
    if constexpr (std::is_same<T, int>()) {
        return reinterpret_cast<const T*>(tensor.data_ptr<int32_t>());
    } else if constexpr (std::is_same<T, long>() || std::is_same<T, long long>()) {
        return reinterpret_cast<const T*>(tensor.data_ptr<int64_t>());
    } else {
        return tensor.data_ptr<T>(); 
    }
}

namespace mc_wrapper {
    template <typename Scalar, typename IndexType>
    class MC_Wrapper {
        mc::MC<Scalar, IndexType> mc;

        static_assert(std::is_same<Scalar, float>() || std::is_same<Scalar, __half>());
        static_assert(std::is_same<IndexType, long>() || std::is_same<IndexType, long long>() || std::is_same<IndexType, int>());

    public:

        std::tuple<torch::Tensor, torch::Tensor, std::optional<torch::Tensor>> forward(
            torch::Tensor grid_vertices, // N * 3 array of grid vertex positions
            torch::Tensor cubes, // (N-1) * 8 array of cube vertex indices
            torch::Tensor values, // N array of scalar values at grid vertices
            Scalar iso,
            std::optional<torch::Tensor> grid_colors = std::nullopt // N * 3 array of grid vertex colors (optional)
        ) {
            CHECK_INPUT(grid_vertices);
            CHECK_INPUT(cubes);
            CHECK_INPUT(values);

            IndexType n_cubes = cubes.size(0);
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

            const auto* grid_vertices_ptr = reinterpret_cast<primitive::Vertex<Scalar> const *>(get_tensor_ptr_const<Scalar>(grid_vertices));
            const auto* cubes_ptr = get_tensor_ptr_const<IndexType>(cubes);
            const auto* values_ptr = get_tensor_ptr_const<Scalar>(values);

            const primitive::Vertex<Scalar>* grid_colors_ptr = nullptr;
            if (grid_colors.has_value()) {
                grid_colors = grid_colors->contiguous(); // ensure contiguous memory 
                CHECK_INPUT(*grid_colors);
                TORCH_INTERNAL_ASSERT(grid_colors->dtype() == scalarType, "colors type must match the mc class");
                grid_colors_ptr = reinterpret_cast<const primitive::Vertex<Scalar>*>(get_tensor_ptr_const<Scalar>(*grid_colors));
            }

            mc.forward(
                grid_vertices_ptr,
                grid_colors_ptr,
                cubes_ptr,
                values_ptr,
                n_cubes,
                iso,
                device // device ID
            );

            IndexType n_verts = mc.n_verts;
            IndexType n_tris = mc.n_tris / 3;

            auto options_float = torch::TensorOptions().dtype(scalarType).device(grid_vertices.device());
            auto options_int = torch::TensorOptions().dtype(indexType).device(cubes.device());

            torch::Tensor verts_tensor = torch::from_blob(
                mc.verts, {n_verts, 3}, options_float
            ).clone();

            torch::Tensor tris_tensor = torch::from_blob(
                mc.tris, {n_tris, 3}, options_int
            ).clone();

            std::optional<torch::Tensor> out_colors_tensor = std::nullopt;
            if (grid_colors.has_value() && mc.out_colors != nullptr && n_verts > 0) {
                out_colors_tensor = torch::from_blob(
                    mc.out_colors, {n_verts, 3}, options_float
                ).clone();
            }

            return std::make_tuple(verts_tensor, tris_tensor, out_colors_tensor);
        }

        void backward(
            torch::Tensor grid_vertices,
            torch::Tensor values,
            torch::Tensor adj_verts,
            torch::Tensor adj_values,
            Scalar iso,
            std::optional<torch::Tensor> grid_colors = std::nullopt,
            std::optional<torch::Tensor> adj_colors = std::nullopt,
            std::optional<torch::Tensor> adj_grid_colors = std::nullopt
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

            const auto* grid_vertices_ptr = reinterpret_cast<primitive::Vertex<Scalar> const *>(get_tensor_ptr_const<Scalar>(grid_vertices));
            const auto* values_ptr = get_tensor_ptr_const<Scalar>(values);
            const auto* adj_verts_ptr = reinterpret_cast<primitive::Vertex<Scalar> const *>(get_tensor_ptr_const<Scalar>(adj_verts));
            auto* adj_values_ptr = get_tensor_ptr<Scalar>(adj_values);

            const primitive::Vertex<Scalar>* grid_colors_ptr = nullptr;
            const primitive::Vertex<Scalar>* adj_colors_ptr = nullptr;
            primitive::Vertex<Scalar>* adj_grid_colors_ptr = nullptr;

            if (grid_colors.has_value() && adj_colors.has_value() && adj_grid_colors.has_value()) {
                grid_colors = grid_colors->contiguous();
                adj_colors = adj_colors->contiguous();
                adj_grid_colors = adj_grid_colors->contiguous();

                CHECK_INPUT(*grid_colors);
                CHECK_INPUT(*adj_colors);
                CHECK_INPUT(*adj_grid_colors);

                TORCH_INTERNAL_ASSERT(grid_colors->dtype() == scalarType, "grid_colors type mismatch");
                TORCH_INTERNAL_ASSERT(adj_colors->dtype() == scalarType, "adj_colors type mismatch");
                TORCH_INTERNAL_ASSERT(adj_grid_colors->dtype() == scalarType, "adj_grid_colors type mismatch");

                grid_colors_ptr = reinterpret_cast<const primitive::Vertex<Scalar>*>(get_tensor_ptr_const<Scalar>(*grid_colors));
                adj_colors_ptr = reinterpret_cast<const primitive::Vertex<Scalar>*>(get_tensor_ptr_const<Scalar>(*adj_colors));
                adj_grid_colors_ptr = reinterpret_cast<primitive::Vertex<Scalar>*>(get_tensor_ptr<Scalar>(*adj_grid_colors));
            }
            
            if (mc.n_verts == 0) {
                // No vertices were generated, so we can skip the backward pass
                return;
            }

            mc.backward(
                grid_vertices_ptr,
                grid_colors_ptr,
                values_ptr,
                adj_verts_ptr,
                adj_colors_ptr,
                adj_values_ptr,
                adj_grid_colors_ptr,
                iso,
                device // device ID
            );
        }
    };
}

namespace grid_wrapper {
    template <typename Scalar, typename IndexType>
    std::tuple<torch::Tensor, torch::Tensor> pc_to_voxel_grid(
        torch::Tensor points,
        IndexType res_x,
        IndexType res_y,
        IndexType res_z,
        int k_threshold,
        int num_keep,
        float rmi_x, float rmi_y, float rmi_z, float rma_x, float rma_y, float rma_z
    ) {
        CHECK_INPUT(points);
        IndexType num_points = points.size(0);
        int device = points.device().index();

        torch::ScalarType scalarType;   
        if constexpr (std::is_same<Scalar, float>()) {
            scalarType = torch::kFloat;
        } else {
            scalarType = torch::kHalf;
        }
        TORCH_INTERNAL_ASSERT(points.dtype() == scalarType, "points type must match the passed template type");

        torch::ScalarType indexType;
        if constexpr (std::is_same<IndexType, int>()) {
            indexType = torch::kInt;
        } else {
            indexType = torch::kLong;
        }

        primitive::Vertex<Scalar>* out_vertices = nullptr;
        IndexType* out_cubes = nullptr;
        IndexType out_num_vertices = 0;
        IndexType out_num_cubes = 0;

        grid::pc_to_voxel_grid<Scalar, IndexType>(
            reinterpret_cast<primitive::Vertex<Scalar> const *>(get_tensor_ptr_const<Scalar>(points)),
            num_points,
            res_x,
            res_y,
            res_z,
            k_threshold,
            num_keep,
            rmi_x, rmi_y, rmi_z, rma_x, rma_y, rma_z,
            &out_vertices,
            &out_num_vertices,
            &out_cubes,
            &out_num_cubes,
            device
        );

        auto options_float = torch::TensorOptions().dtype(scalarType).device(points.device());
        auto options_int = torch::TensorOptions().dtype(indexType).device(points.device());

        torch::Tensor verts_tensor = torch::empty({0}, options_float);
        torch::Tensor cubes_tensor = torch::empty({0}, options_int);

        if (out_num_vertices > 0 && out_vertices != nullptr) {
            verts_tensor = torch::from_blob(out_vertices, {out_num_vertices, 3}, options_float).clone();
            cudaFree(out_vertices);
        }

        if (out_num_cubes > 0 && out_cubes != nullptr) {
            cubes_tensor = torch::from_blob(out_cubes, {out_num_cubes, 8}, options_int).clone();
            cudaFree(out_cubes);
        }

        return std::make_tuple(verts_tensor, cubes_tensor);
    }

    template <typename Scalar, typename IndexType>
    std::tuple<torch::Tensor, torch::Tensor> pc_to_voxel_grid_chunk(
        torch::Tensor points,
        IndexType res_x,
        IndexType res_y,
        IndexType res_z,
        int chunk_size,
        int k_threshold,
        int num_keep,
        float rmi_x, float rmi_y, float rmi_z, float rma_x, float rma_y, float rma_z
    ) {
        CHECK_INPUT(points);
        IndexType num_points = points.size(0);
        int device = points.device().index();

        torch::ScalarType scalarType;   
        if constexpr (std::is_same<Scalar, float>()) {
            scalarType = torch::kFloat;
        } else {
            scalarType = torch::kHalf;
        }
        TORCH_INTERNAL_ASSERT(points.dtype() == scalarType, "points type must match the passed template type");

        torch::ScalarType indexType;
        if constexpr (std::is_same<IndexType, int>()) {
            indexType = torch::kInt;
        } else {
            indexType = torch::kLong;
        }

        primitive::Vertex<Scalar>* out_vertices = nullptr;
        IndexType* out_cubes = nullptr;
        IndexType out_num_vertices = 0;
        IndexType out_num_cubes = 0;

        grid::pc_to_voxel_grid_chunk<Scalar, IndexType>(
            reinterpret_cast<primitive::Vertex<Scalar> const *>(get_tensor_ptr_const<Scalar>(points)),
            num_points,
            res_x,
            res_y,
            res_z,
            chunk_size,
            k_threshold,
            num_keep,
            rmi_x, rmi_y, rmi_z, rma_x, rma_y, rma_z,
            &out_vertices,
            &out_num_vertices,
            &out_cubes,
            &out_num_cubes,
            device
        );

        auto options_float = torch::TensorOptions().dtype(scalarType).device(points.device());
        auto options_int = torch::TensorOptions().dtype(indexType).device(points.device());

        torch::Tensor verts_tensor = torch::empty({0}, options_float);
        torch::Tensor cubes_tensor = torch::empty({0}, options_int);

        if (out_num_vertices > 0 && out_vertices != nullptr) {
            verts_tensor = torch::from_blob(out_vertices, {out_num_vertices, 3}, options_float).clone();
            cudaFree(out_vertices);
        }

        if (out_num_cubes > 0 && out_cubes != nullptr) {
            cubes_tensor = torch::from_blob(out_cubes, {out_num_cubes, 8}, options_int).clone();
            cudaFree(out_cubes);
        }

        return std::make_tuple(verts_tensor, cubes_tensor);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // ==========================================
    // 32-bit versions (int)
    // ==========================================
    pybind11::class_<mc_wrapper::MC_Wrapper<float, int>>(m, "MCFI")
        .def(pybind11::init<>())
        .def("forward", &mc_wrapper::MC_Wrapper<float, int>::forward, 
            py::arg("grid_vertices"), py::arg("cubes"), py::arg("values"), py::arg("iso"), py::arg("grid_colors") = py::none())
        .def("backward", &mc_wrapper::MC_Wrapper<float, int>::backward,
            py::arg("grid_vertices"), py::arg("values"), py::arg("adj_verts"), py::arg("adj_values"), py::arg("iso"), 
            py::arg("grid_colors") = py::none(), py::arg("adj_colors") = py::none(), py::arg("adj_grid_colors") = py::none());
    
    m.def("pc_to_voxel_grid", &grid_wrapper::pc_to_voxel_grid<float, int>, 
        "Convert a point cloud to a sparse voxel grid.",
        pybind11::arg("points"), pybind11::arg("res_x"), pybind11::arg("res_y"), pybind11::arg("res_z"), 
        pybind11::arg("k_threshold"), pybind11::arg("num_keep"), 
        pybind11::arg("rmi_x"), pybind11::arg("rmi_y"), pybind11::arg("rmi_z"), 
        pybind11::arg("rma_x"), pybind11::arg("rma_y"), pybind11::arg("rma_z"));

    m.def("pc_to_voxel_grid_chunk", &grid_wrapper::pc_to_voxel_grid_chunk<float, int>, 
        "Convert a point cloud to a sparse voxel grid using spatial chunking.",
        pybind11::arg("points"), pybind11::arg("res_x"), pybind11::arg("res_y"), pybind11::arg("res_z"), 
        pybind11::arg("chunk_size"),
        pybind11::arg("k_threshold"), pybind11::arg("num_keep"), 
        pybind11::arg("rmi_x"), pybind11::arg("rmi_y"), pybind11::arg("rmi_z"), 
        pybind11::arg("rma_x"), pybind11::arg("rma_y"), pybind11::arg("rma_z"));


    // ==========================================
    // 64-bit versions (long long)
    // ==========================================
    pybind11::class_<mc_wrapper::MC_Wrapper<float, long long>>(m, "MCFL")
        .def(pybind11::init<>())
        .def("forward", &mc_wrapper::MC_Wrapper<float, long long>::forward,
            py::arg("grid_vertices"), py::arg("cubes"), py::arg("values"), py::arg("iso"), py::arg("grid_colors") = py::none())
        .def("backward", &mc_wrapper::MC_Wrapper<float, long long>::backward,
            py::arg("grid_vertices"), py::arg("values"), py::arg("adj_verts"), py::arg("adj_values"), py::arg("iso"), 
            py::arg("grid_colors") = py::none(), py::arg("adj_colors") = py::none(), py::arg("adj_grid_colors") = py::none());

    m.def("pc_to_voxel_grid_long", &grid_wrapper::pc_to_voxel_grid<float, long long>, 
        "Convert a massive point cloud to a sparse voxel grid using 64-bit indices.",
        pybind11::arg("points"), pybind11::arg("res_x"), pybind11::arg("res_y"), pybind11::arg("res_z"), 
        pybind11::arg("k_threshold"), pybind11::arg("num_keep"), 
        pybind11::arg("rmi_x"), pybind11::arg("rmi_y"), pybind11::arg("rmi_z"), 
        pybind11::arg("rma_x"), pybind11::arg("rma_y"), pybind11::arg("rma_z"));

    m.def("pc_to_voxel_grid_chunk_long", &grid_wrapper::pc_to_voxel_grid_chunk<float, long long>, 
        "Convert a massive point cloud to a sparse voxel grid using spatial chunking and 64-bit indices.",
        pybind11::arg("points"), pybind11::arg("res_x"), pybind11::arg("res_y"), pybind11::arg("res_z"), 
        pybind11::arg("chunk_size"),
        pybind11::arg("k_threshold"), pybind11::arg("num_keep"), 
        pybind11::arg("rmi_x"), pybind11::arg("rmi_y"), pybind11::arg("rmi_z"), 
        pybind11::arg("rma_x"), pybind11::arg("rma_y"), pybind11::arg("rma_z"));
}