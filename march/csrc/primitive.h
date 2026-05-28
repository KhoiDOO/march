#ifndef PRIMITIVE_H
#define PRIMITIVE_H

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

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

namespace primitive
{

    template <typename T>
    struct Vertex
    {
        T x, y, z;

        inline __device__ __host__ T *data_ptr() { return &x; }

        inline __device__ __host__ Vertex<T> operator+(Vertex<T> const &other) const
        {
            return {x + other.x, y + other.y, z + other.z};
        }
        inline __device__ __host__ T dot(Vertex<T> const &other) const
        {
            return x * other.x + y * other.y + z * other.z;
        }

        inline __device__ __host__ Vertex<T> operator-(Vertex<T> const &other) const
        {
            return {x - other.x, y - other.y, z - other.z};
        }

        inline __device__ __host__ Vertex<T> operator*(Vertex<T> const &other) const
        {
            return {x * other.x, y * other.y, z * other.z};
        }

        inline __device__ __host__ Vertex<T> operator*(T const &scalar) const
        {
            return {x * scalar, y * scalar, z * scalar};
        }

        inline __device__ __host__ Vertex<T> operator/(T const &scalar) const
        {
            return {x / scalar, y / scalar, z / scalar};
        }

        inline __device__ __host__ Vertex<T> &operator+=(Vertex<T> const &other)
        {
            x += other.x;
            y += other.y;
            z += other.z;
            return *this;
        }

        inline __device__ __host__ Vertex<T> &operator-=(Vertex<T> const &other)
        {
            x -= other.x;
            y -= other.y;
            z -= other.z;
            return *this;
        }

        inline __device__ __host__ Vertex<T> &operator*=(T const &scalar)
        {
            x *= scalar;
            y *= scalar;
            z *= scalar;
            return *this;
        }
    };

    template <typename T>
    struct Triangle
    {
        T i, j, k;
        inline __device__ __host__ T *data_ptr() { return &i; }
    };

    template <typename T>
    struct Quad
    {
        T i, j, k, l;
        inline __device__ __host__ T *data_ptr() { return &i; }
    };

    template <typename T>
    struct BBox
    {
        Vertex<T> min_pt, max_pt;

        // A helper to initialize a BBox from a single point
        inline static __device__ __host__ BBox<T> from_point(const Vertex<T> &pt)
        {
            return {pt, pt};
        }

        // A helper to merge two bounding boxes
        inline __device__ __host__ BBox<T> merge(const BBox<T> &other) const
        {
            BBox<T> res;
            res.min_pt.x = min(min_pt.x, other.min_pt.x);
            res.min_pt.y = min(min_pt.y, other.min_pt.y);
            res.min_pt.z = min(min_pt.z, other.min_pt.z);

            res.max_pt.x = max(max_pt.x, other.max_pt.x);
            res.max_pt.y = max(max_pt.y, other.max_pt.y);
            res.max_pt.z = max(max_pt.z, other.max_pt.z);
            return res;
        }
    };

    template <typename IndexType>
    struct EdgeKey
    {
        IndexType v0;
        IndexType v1;

        // Required by thrust::sort and binary search
        __host__ __device__ bool operator<(const EdgeKey &other) const
        {
            if (v0 != other.v0)
                return v0 < other.v0;
            return v1 < other.v1;
        }

        // Required by thrust::unique
        __host__ __device__ bool operator==(const EdgeKey &other) const
        {
            return v0 == other.v0 && v1 == other.v1;
        }
    };

} // namespace primitive

#endif // PRIMITIVE_H