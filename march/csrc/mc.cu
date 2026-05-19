#include "mc.h"
#include "primitive.h"

#include <cstdint>
#include <cstdio>

#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/unique.h>
#include <thrust/binary_search.h>
#include <thrust/iterator/transform_iterator.h>

namespace mc {
    bool check_cuda_result(cudaError_t code, const char *file, int line)
    {
        if (code == cudaSuccess) return true;

        fprintf(stderr, "CUDA error %u: %s (%s:%d)\n", unsigned(code), cudaGetErrorString(code), file, line);
        return false;
    }

    #define CHECK_CUDA(code) check_cuda_result((code), __FILE__, __LINE__)

    template <typename T>
    inline __device__ __host__ T min(T a, T b) { return a < b ? a : b; }

    template <typename T>
    inline __device__ __host__ T max(T a, T b) { return a > b ? a : b; }
    
    template <typename T>
    inline __device__ __host__ T clamp(T x, T a, T b){ return min(max(a, x), b); }

    // constexpr int BLOCK_SIZE = 512;

    __constant__ int edge2vertices[12][2] = {
        {0, 1},
        {1, 5},
        {4, 5},
        {0, 4},
        {2, 3},
        {3, 7},
        {6, 7},
        {2, 6},
        {0, 2},
        {1, 3},
        {5, 7},
        {4, 6}
    };

    __constant__ int firstMarchingCubesId[257] = {
        0, 0, 3, 6, 12, 15, 21, 27, 36, 39, 45, 51, 60, 66, 75, 84, 90, 93, 99, 105, 114,
        120, 129, 138, 150, 156, 165, 174, 186, 195, 207, 219, 228, 231, 237, 243, 252, 258, 267, 276, 288,
        294, 303, 312, 324, 333, 345, 357, 366, 372, 381, 390, 396, 405, 417, 429, 438, 447, 459, 471, 480,
        492, 507, 522, 528, 531, 537, 543, 552, 558, 567, 576, 588, 594, 603, 612, 624, 633, 645, 657, 666,
        672, 681, 690, 702, 711, 723, 735, 750, 759, 771, 783, 798, 810, 825, 840, 852, 858, 867, 876, 888,
        897, 909, 915, 924, 933, 945, 957, 972, 984, 999, 1008, 1014, 1023, 1035, 1047, 1056, 1068, 1083, 1092, 1098,
        1110, 1125, 1140, 1152, 1167, 1173, 1185, 1188, 1191, 1197, 1203, 1212, 1218, 1227, 1236, 1248, 1254, 1263, 1272, 1284,
        1293, 1305, 1317, 1326, 1332, 1341, 1350, 1362, 1371, 1383, 1395, 1410, 1419, 1425, 1437, 1446, 1458, 1467, 1482, 1488,
        1494, 1503, 1512, 1524, 1533, 1545, 1557, 1572, 1581, 1593, 1605, 1620, 1632, 1647, 1662, 1674, 1683, 1695, 1707, 1716,
        1728, 1743, 1758, 1770, 1782, 1791, 1806, 1812, 1827, 1839, 1845, 1848, 1854, 1863, 1872, 1884, 1893, 1905, 1917, 1932,
        1941, 1953, 1965, 1980, 1986, 1995, 2004, 2010, 2019, 2031, 2043, 2058, 2070, 2085, 2100, 2106, 2118, 2127, 2142, 2154,
        2163, 2169, 2181, 2184, 2193, 2205, 2217, 2232, 2244, 2259, 2268, 2280, 2292, 2307, 2322, 2328, 2337, 2349, 2355, 2358,
        2364, 2373, 2382, 2388, 2397, 2409, 2415, 2418, 2427, 2433, 2445, 2448, 2454, 2457, 2460, 2460};
    
    int h_firstMarchingCubesId[257] = {
        0, 0, 3, 6, 12, 15, 21, 27, 36, 39, 45, 51, 60, 66, 75, 84, 90, 93, 99, 105, 114,
        120, 129, 138, 150, 156, 165, 174, 186, 195, 207, 219, 228, 231, 237, 243, 252, 258, 267, 276, 288,
        294, 303, 312, 324, 333, 345, 357, 366, 372, 381, 390, 396, 405, 417, 429, 438, 447, 459, 471, 480,
        492, 507, 522, 528, 531, 537, 543, 552, 558, 567, 576, 588, 594, 603, 612, 624, 633, 645, 657, 666,
        672, 681, 690, 702, 711, 723, 735, 750, 759, 771, 783, 798, 810, 825, 840, 852, 858, 867, 876, 888,
        897, 909, 915, 924, 933, 945, 957, 972, 984, 999, 1008, 1014, 1023, 1035, 1047, 1056, 1068, 1083, 1092, 1098,
        1110, 1125, 1140, 1152, 1167, 1173, 1185, 1188, 1191, 1197, 1203, 1212, 1218, 1227, 1236, 1248, 1254, 1263, 1272, 1284,
        1293, 1305, 1317, 1326, 1332, 1341, 1350, 1362, 1371, 1383, 1395, 1410, 1419, 1425, 1437, 1446, 1458, 1467, 1482, 1488,
        1494, 1503, 1512, 1524, 1533, 1545, 1557, 1572, 1581, 1593, 1605, 1620, 1632, 1647, 1662, 1674, 1683, 1695, 1707, 1716,
        1728, 1743, 1758, 1770, 1782, 1791, 1806, 1812, 1827, 1839, 1845, 1848, 1854, 1863, 1872, 1884, 1893, 1905, 1917, 1932,
        1941, 1953, 1965, 1980, 1986, 1995, 2004, 2010, 2019, 2031, 2043, 2058, 2070, 2085, 2100, 2106, 2118, 2127, 2142, 2154,
        2163, 2169, 2181, 2184, 2193, 2205, 2217, 2232, 2244, 2259, 2268, 2280, 2292, 2307, 2322, 2328, 2337, 2349, 2355, 2358,
        2364, 2373, 2382, 2388, 2397, 2409, 2415, 2418, 2427, 2433, 2445, 2448, 2454, 2457, 2460, 2460};
    
    __constant__ int marchingCubesIds[2460] = {
        0, 8, 3,
        0, 1, 9,
        1, 8, 3, 9, 8, 1,
        1, 2, 10,
        0, 8, 3, 1, 2, 10,
        9, 2, 10, 0, 2, 9,
        2, 8, 3, 2, 10, 8, 10, 9, 8,
        3, 11, 2,
        0, 11, 2, 8, 11, 0,
        1, 9, 0, 2, 3, 11,
        1, 11, 2, 1, 9, 11, 9, 8, 11,
        3, 10, 1, 11, 10, 3,
        0, 10, 1, 0, 8, 10, 8, 11, 10,
        3, 9, 0, 3, 11, 9, 11, 10, 9,
        9, 8, 10, 10, 8, 11,
        4, 7, 8,
        4, 3, 0, 7, 3, 4,
        0, 1, 9, 8, 4, 7,
        4, 1, 9, 4, 7, 1, 7, 3, 1,
        1, 2, 10, 8, 4, 7,
        3, 4, 7, 3, 0, 4, 1, 2, 10,
        9, 2, 10, 9, 0, 2, 8, 4, 7,
        2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4,
        8, 4, 7, 3, 11, 2,
        11, 4, 7, 11, 2, 4, 2, 0, 4,
        9, 0, 1, 8, 4, 7, 2, 3, 11,
        4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1,
        3, 10, 1, 3, 11, 10, 7, 8, 4,
        1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4,
        4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3,
        4, 7, 11, 4, 11, 9, 9, 11, 10,
        9, 5, 4,
        9, 5, 4, 0, 8, 3,
        0, 5, 4, 1, 5, 0,
        8, 5, 4, 8, 3, 5, 3, 1, 5,
        1, 2, 10, 9, 5, 4,
        3, 0, 8, 1, 2, 10, 4, 9, 5,
        5, 2, 10, 5, 4, 2, 4, 0, 2,
        2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8,
        9, 5, 4, 2, 3, 11,
        0, 11, 2, 0, 8, 11, 4, 9, 5,
        0, 5, 4, 0, 1, 5, 2, 3, 11,
        2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5,
        10, 3, 11, 10, 1, 3, 9, 5, 4,
        4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10,
        5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3,
        5, 4, 8, 5, 8, 10, 10, 8, 11,
        9, 7, 8, 5, 7, 9,
        9, 3, 0, 9, 5, 3, 5, 7, 3,
        0, 7, 8, 0, 1, 7, 1, 5, 7,
        1, 5, 3, 3, 5, 7,
        9, 7, 8, 9, 5, 7, 10, 1, 2,
        10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3,
        8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2,
        2, 10, 5, 2, 5, 3, 3, 5, 7,
        7, 9, 5, 7, 8, 9, 3, 11, 2,
        9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11,
        2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7,
        11, 2, 1, 11, 1, 7, 7, 1, 5,
        9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11,
        5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0,
        11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0,
        11, 10, 5, 7, 11, 5,
        10, 6, 5,
        0, 8, 3, 5, 10, 6,
        9, 0, 1, 5, 10, 6,
        1, 8, 3, 1, 9, 8, 5, 10, 6,
        1, 6, 5, 2, 6, 1,
        1, 6, 5, 1, 2, 6, 3, 0, 8,
        9, 6, 5, 9, 0, 6, 0, 2, 6,
        5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8,
        2, 3, 11, 10, 6, 5,
        11, 0, 8, 11, 2, 0, 10, 6, 5,
        0, 1, 9, 2, 3, 11, 5, 10, 6,
        5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11,
        6, 3, 11, 6, 5, 3, 5, 1, 3,
        0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6,
        3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9,
        6, 5, 9, 6, 9, 11, 11, 9, 8,
        5, 10, 6, 4, 7, 8,
        4, 3, 0, 4, 7, 3, 6, 5, 10,
        1, 9, 0, 5, 10, 6, 8, 4, 7,
        10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4,
        6, 1, 2, 6, 5, 1, 4, 7, 8,
        1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7,
        8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6,
        7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9,
        3, 11, 2, 7, 8, 4, 10, 6, 5,
        5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11,
        0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6,
        9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6,
        8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6,
        5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11,
        0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7,
        6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9,
        10, 4, 9, 6, 4, 10,
        4, 10, 6, 4, 9, 10, 0, 8, 3,
        10, 0, 1, 10, 6, 0, 6, 4, 0,
        8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10,
        1, 4, 9, 1, 2, 4, 2, 6, 4,
        3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4,
        0, 2, 4, 4, 2, 6,
        8, 3, 2, 8, 2, 4, 4, 2, 6,
        10, 4, 9, 10, 6, 4, 11, 2, 3,
        0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6,
        3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10,
        6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1,
        9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3,
        8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1,
        3, 11, 6, 3, 6, 0, 0, 6, 4,
        6, 4, 8, 11, 6, 8,
        7, 10, 6, 7, 8, 10, 8, 9, 10,
        0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10,
        10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0,
        10, 6, 7, 10, 7, 1, 1, 7, 3,
        1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7,
        2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9,
        7, 8, 0, 7, 0, 6, 6, 0, 2,
        7, 3, 2, 6, 7, 2,
        2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7,
        2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7,
        1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11,
        11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1,
        8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6,
        0, 9, 1, 11, 6, 7,
        7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0,
        7, 11, 6,
        7, 6, 11,
        3, 0, 8, 11, 7, 6,
        0, 1, 9, 11, 7, 6,
        8, 1, 9, 8, 3, 1, 11, 7, 6,
        10, 1, 2, 6, 11, 7,
        1, 2, 10, 3, 0, 8, 6, 11, 7,
        2, 9, 0, 2, 10, 9, 6, 11, 7,
        6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8,
        7, 2, 3, 6, 2, 7,
        7, 0, 8, 7, 6, 0, 6, 2, 0,
        2, 7, 6, 2, 3, 7, 0, 1, 9,
        1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6,
        10, 7, 6, 10, 1, 7, 1, 3, 7,
        10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8,
        0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7,
        7, 6, 10, 7, 10, 8, 8, 10, 9,
        6, 8, 4, 11, 8, 6,
        3, 6, 11, 3, 0, 6, 0, 4, 6,
        8, 6, 11, 8, 4, 6, 9, 0, 1,
        9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6,
        6, 8, 4, 6, 11, 8, 2, 10, 1,
        1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6,
        4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9,
        10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3,
        8, 2, 3, 8, 4, 2, 4, 6, 2,
        0, 4, 2, 4, 6, 2,
        1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8,
        1, 9, 4, 1, 4, 2, 2, 4, 6,
        8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1,
        10, 1, 0, 10, 0, 6, 6, 0, 4,
        4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3,
        10, 9, 4, 6, 10, 4,
        4, 9, 5, 7, 6, 11,
        0, 8, 3, 4, 9, 5, 11, 7, 6,
        5, 0, 1, 5, 4, 0, 7, 6, 11,
        11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5,
        9, 5, 4, 10, 1, 2, 7, 6, 11,
        6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5,
        7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2,
        3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6,
        7, 2, 3, 7, 6, 2, 5, 4, 9,
        9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7,
        3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0,
        6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8,
        9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7,
        1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4,
        4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10,
        7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10,
        6, 9, 5, 6, 11, 9, 11, 8, 9,
        3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5,
        0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11,
        6, 11, 3, 6, 3, 5, 5, 3, 1,
        1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6,
        0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10,
        11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5,
        6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3,
        5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2,
        9, 5, 6, 9, 6, 0, 0, 6, 2,
        1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8,
        1, 5, 6, 2, 1, 6,
        1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6,
        10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0,
        0, 3, 8, 5, 6, 10,
        10, 5, 6,
        11, 5, 10, 7, 5, 11,
        11, 5, 10, 11, 7, 5, 8, 3, 0,
        5, 11, 7, 5, 10, 11, 1, 9, 0,
        10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1,
        11, 1, 2, 11, 7, 1, 7, 5, 1,
        0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11,
        9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7,
        7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2,
        2, 5, 10, 2, 3, 5, 3, 7, 5,
        8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5,
        9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2,
        9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2,
        1, 3, 5, 3, 7, 5,
        0, 8, 7, 0, 7, 1, 1, 7, 5,
        9, 0, 3, 9, 3, 5, 5, 3, 7,
        9, 8, 7, 5, 9, 7,
        5, 8, 4, 5, 10, 8, 10, 11, 8,
        5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0,
        0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5,
        10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4,
        2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8,
        0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11,
        0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5,
        9, 4, 5, 2, 11, 3,
        2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4,
        5, 10, 2, 5, 2, 4, 4, 2, 0,
        3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9,
        5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2,
        8, 4, 5, 8, 5, 3, 3, 5, 1,
        0, 4, 5, 1, 0, 5,
        8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5,
        9, 4, 5,
        4, 11, 7, 4, 9, 11, 9, 10, 11,
        0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11,
        1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11,
        3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4,
        4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2,
        9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3,
        11, 7, 4, 11, 4, 2, 2, 4, 0,
        11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4,
        2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9,
        9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7,
        3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10,
        1, 10, 2, 8, 7, 4,
        4, 9, 1, 4, 1, 7, 7, 1, 3,
        4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1,
        4, 0, 3, 7, 4, 3,
        4, 8, 7,
        9, 10, 8, 10, 11, 8,
        3, 0, 9, 3, 9, 11, 11, 9, 10,
        0, 1, 10, 0, 10, 8, 8, 10, 11,
        3, 1, 10, 11, 3, 10,
        1, 2, 11, 1, 11, 9, 9, 11, 8,
        3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9,
        0, 2, 11, 8, 0, 11,
        3, 2, 11,
        2, 3, 8, 2, 8, 10, 10, 8, 9,
        9, 10, 2, 0, 9, 2,
        2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8,
        1, 10, 2,
        1, 3, 8, 9, 1, 8,
        0, 9, 1,
        0, 3, 8,
    };

    __constant__ int index[8] = {0, 1, 5, 4, 2, 3, 7, 6};

    struct IsActiveOp {
        __host__ __device__
        int operator()(const uint8_t code) const {
            return (code > 0 && code < 255) ? 1 : 0;
        }
    };

    struct TriCountOp {
        __device__
        int operator()(const uint8_t code) const {
            // Find the length using your standard offset table
            return firstMarchingCubesId[code + 1] - firstMarchingCubesId[code];
        }
    };

    template <typename Scalar, typename IndexType>
    __global__ void identify_active_cubes_kernel(
        const IndexType* cubes,
        const Scalar* values,
        IndexType n_cubes,
        Scalar iso,
        uint8_t* cube_codes
    ) {
        int cube_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (cube_idx >= n_cubes) return;

        IndexType const *v_ptr = &cubes[cube_idx * 8]; // 8 vertex indices of the cube
        uint8_t code = 0;
        for (int i = 0; i < 8; ++i) {
            if (values[v_ptr[index[i]]] >= iso) {
                code |= (1 << i);
            }
        }
        cube_codes[cube_idx] = (uint8_t)code;
    };

    template <typename IndexType>
    __global__ void compact_active_cubes_kernel(
        const uint8_t* cube_codes,
        const IndexType* prefix_sum,
        IndexType n_cubes,
        IndexType* used_cube_index,
        uint8_t* used_cube_code
    ) {
        int cube_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (cube_idx >= n_cubes) return;

        uint8_t code = cube_codes[cube_idx];

        if (code > 0 && code < 255) { // Active cube

            IndexType pos = prefix_sum[cube_idx];
            used_cube_index[pos] = cube_idx;
            used_cube_code[pos] = code;
        }
    };

    template <typename IndexType>
    __global__ void extract_active_edges_kernel(
        const IndexType *cubes,
        const IndexType *used_cube_index,
        const uint8_t* used_cube_code,
        long long* active_edges,
        IndexType n_used_cubes
    ) {
        int active_cube_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (active_cube_idx >= n_used_cubes) return;

        IndexType cube_idx = used_cube_index[active_cube_idx];
        uint8_t code = used_cube_code[active_cube_idx];
        const IndexType *v_ptr = &cubes[cube_idx * 8]; // 8 vertex indices of the cube

        int face_start = firstMarchingCubesId[code];
        int face_num = firstMarchingCubesId[code + 1] - face_start;

        for (int i = 0; i < 12; i++) {
            active_edges[active_cube_idx * 12 + i] = -1LL;
        }

        for (int i = 0; i < face_num; ++i) {

            int local_edge_id = marchingCubesIds[face_start + i];

            IndexType v0 = v_ptr[edge2vertices[local_edge_id][0]];
            IndexType v1 = v_ptr[edge2vertices[local_edge_id][1]];
            active_edges[active_cube_idx * 12 + local_edge_id] = (static_cast<long long>(min(v0, v1)) << 32) | max(v0, v1);
        }
    };

    template <typename IndexType>
    __global__ void build_edge_map_kernel(
        const IndexType* cubes,
        const IndexType* used_cube_index,
        const long long* unique_edges,
        IndexType* cube_edge_to_vert_idx,
        IndexType n_used_cubes,
        IndexType n_verts
    ) {
        int active_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (active_idx >= n_used_cubes) return;

        IndexType global_cube_idx = used_cube_index[active_idx];
        const IndexType *v_ptr = &cubes[global_cube_idx * 8];

        for (int i = 0; i < 12; ++i) {
            IndexType v0 = v_ptr[edge2vertices[i][0]];
            IndexType v1 = v_ptr[edge2vertices[i][1]];
            long long edge_key = (static_cast<long long>(min(v0, v1)) << 32) | max(v0, v1);

            // Binary search in unique_edges to find the vertex index
            IndexType left = 0;
            IndexType right = n_verts - 1;
            IndexType unique_id = -1;

            while (left <= right) {
                IndexType mid = left + (right - left) / 2;
                if (unique_edges[mid] == edge_key) {
                    unique_id = mid;
                    break;
                } else if (unique_edges[mid] < edge_key) {
                    left = mid + 1;
                } else {
                    right = mid - 1;
                }
            }

            cube_edge_to_vert_idx[active_idx * 12 + i] = unique_id; // Store the vertex index for this edge
        }
    };

    template <typename Scalar, typename IndexType>
    __global__ void interpolate_vertices_kernel(
        const long long* unique_edges,
        const Vertex<Scalar>* grid_vertices,
        const Scalar* values,
        IndexType n_verts,
        Scalar iso,
        Vertex<Scalar>* out_verts
    ) {
        int v_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (v_idx >= n_verts) return;

        // 1. Decode the 64-bit edge
        long long edge_sig = unique_edges[v_idx];
        IndexType v0_idx = static_cast<IndexType>(edge_sig >> 32);
        IndexType v1_idx = static_cast<IndexType>(edge_sig & 0xFFFFFFFF);

        // 2. Fetch positions and values
        Vertex<Scalar> p0 = grid_vertices[v0_idx];
        Vertex<Scalar> p1 = grid_vertices[v1_idx];
        Scalar val0 = values[v0_idx];
        Scalar val1 = values[v1_idx];

        // 3. Interpolate (Differentiable formula)
        Scalar t = (val1 != val0) ? clamp((iso - val0) / (val1 - val0), Scalar(0.0), Scalar(1.0)) : Scalar(0.5);

        // Result = P0 + (P1 - P0) * t
        out_verts[v_idx] = p0 + (p1 - p0) * t;
    };

    template <typename IndexType>
    __global__ void assemble_triangles_kernel(
        const uint8_t* used_cube_code,
        const IndexType* tri_prefix_sum,
        const IndexType* cube_edge_to_vert_idx,
        IndexType n_used_cubes,
        IndexType* tris
    ) {
        int active_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (active_idx >= n_used_cubes) return;

        // 1. Get the code for this cube
        uint8_t code = used_cube_code[active_idx];

        // 2. Find the start/count in the FACE_TABLE
        int face_start = firstMarchingCubesId[code];
        int face_num = firstMarchingCubesId[code + 1] - face_start;

        // 3. Find where we start writing in the global 'tris' array
        int out_start = tri_prefix_sum[active_idx];

        // 4. Copy the vertex IDs using the map
        for (int i = 0; i < face_num; i++) {
            int local_edge_id = marchingCubesIds[face_start + i];
            // Use the map to get the shared unique vertex ID
            int unique_v_id = cube_edge_to_vert_idx[active_idx * 12 + local_edge_id];
            tris[out_start + i] = unique_v_id;
        }
    };

    template <typename Scalar, typename IndexType>
    void MC<Scalar, IndexType>::ensure_grid_storage_size(size_t n_cubes) {
        if (n_cubes > this->allocated_cube_count) {
            this->allocated_cube_count = n_cubes + n_cubes / 5; // Add 20% buffer to avoid frequent reallocations

            // Free old memory
            if (this->temp_buffer) CHECK_CUDA(cudaFree(this->temp_buffer));
            if (this->cube_codes) CHECK_CUDA(cudaFree(this->cube_codes));

            // Allocate new memory
            CHECK_CUDA(cudaMalloc((void **)&this->temp_buffer, this->allocated_cube_count * sizeof(IndexType)));
            CHECK_CUDA(cudaMalloc((void **)&this->cube_codes, this->allocated_cube_count * sizeof(uint8_t)));
        }
    };

    template <typename Scalar, typename IndexType>
    void MC<Scalar, IndexType>::ensure_used_cube_storage_size(size_t n_used_cubes) {
        if (n_used_cubes > this->allocated_used_cube_count) {

            this->allocated_used_cube_count = n_used_cubes + n_used_cubes / 5; // Add 20% buffer to avoid frequent reallocations

            // Free old memory
            if (this->used_cube_code) CHECK_CUDA(cudaFree(this->used_cube_code));
            if (this->used_cube_index) CHECK_CUDA(cudaFree(this->used_cube_index));
            if (this->used_to_first_mc_tri) CHECK_CUDA(cudaFree(this->used_to_first_mc_tri));
            if (this->cube_edge_to_vert_idx) CHECK_CUDA(cudaFree(this->cube_edge_to_vert_idx));

            // Allocate new memory
            CHECK_CUDA(cudaMalloc((void **)&this->used_cube_code, this->allocated_used_cube_count * sizeof(uint8_t)));
            CHECK_CUDA(cudaMalloc((void **)&this->used_cube_index, this->allocated_used_cube_count * sizeof(IndexType)));
            CHECK_CUDA(cudaMalloc((void **)&this->used_to_first_mc_tri, this->allocated_used_cube_count * sizeof(IndexType)));
            CHECK_CUDA(cudaMalloc((void **)&this->cube_edge_to_vert_idx, this->allocated_used_cube_count * 12 * sizeof(IndexType)));
        }
    };

    template <typename Scalar, typename IndexType>
    void MC<Scalar, IndexType>::ensure_vert_storage_size(size_t n_verts) {
        if (n_verts > this->allocated_vert_count) {
            
            this->allocated_vert_count = n_verts + n_verts / 5;  // Add 20% buffer to avoid frequent reallocations

            // Free old memory
            if (this->unique_edges) CHECK_CUDA(cudaFree(this->unique_edges));
            if (this->verts) CHECK_CUDA(cudaFree(this->verts));

            // Allocate new memory
            CHECK_CUDA(cudaMalloc((void **)&this->unique_edges, this->allocated_vert_count * sizeof(long long)));
            CHECK_CUDA(cudaMalloc((void **)&this->verts, this->allocated_vert_count * sizeof(Vertex<Scalar>)));
        }
    };

    template <typename Scalar, typename IndexType>
    void MC<Scalar, IndexType>::ensure_tri_storage_size(size_t n_tris) {
        if (n_tris > this->allocated_tri_count) {
            this->allocated_tri_count = n_tris + n_tris / 5;  // Add 20% buffer to avoid frequent reallocations

            // Free old memory
            if (this->tris) CHECK_CUDA(cudaFree(this->tris));

            // Allocate new memory
            CHECK_CUDA(cudaMalloc((void **)&this->tris, this->allocated_tri_count * sizeof(IndexType)));
        }
    };

    template <typename Scalar, typename IndexType>
    void MC<Scalar, IndexType>::forward(
        Vertex<Scalar> const *grid_vertices,
        IndexType const *cubes,
        Scalar const *values,
        IndexType n_cubes, 
        Scalar iso, 
        int device
    ) {

        int threads = 256;
        int blocks = (n_cubes + threads - 1) / threads;
        cudaSetDevice(device);
        
        // 0. Ensure we have enough storage for the cube codes and prefix sums
        this->ensure_grid_storage_size(n_cubes);

        // 1. Run your Identification Kernel
        identify_active_cubes_kernel<<<blocks, threads>>>(cubes, values, n_cubes, iso, this->cube_codes);
        CHECK_CUDA(cudaDeviceSynchronize());

        // 2. Wrap the raw 'cube_codes' pointer in a Thrust pointer
        thrust::device_ptr<uint8_t> d_codes(this->cube_codes);

        // 3. Create the "Virtual" iterator
        auto active_flag_iter = thrust::make_transform_iterator(d_codes, IsActiveOp());

        // 4. Prefix Sum Output
        thrust::device_ptr<IndexType> d_prefix_sum(this->temp_buffer);

        // 5. Run Exclusive Scan (Prefix Sum) on the "Virtual" iterator
        thrust::exclusive_scan(active_flag_iter, active_flag_iter + n_cubes, d_prefix_sum);

        // 6. Get the total number of active cubes from the last element of the prefix sum + last cube's active flag
        uint8_t last_flag;
        IndexType last_sum;
        CHECK_CUDA(cudaMemcpy(&last_flag, this->cube_codes + n_cubes - 1, sizeof(uint8_t), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(&last_sum, this->temp_buffer + n_cubes - 1, sizeof(IndexType), cudaMemcpyDeviceToHost));
        this->n_used_cubes = last_sum + ((last_flag > 0 && last_flag < 255) ? 1 : 0);

        if (this->n_used_cubes == 0) {
            this->n_verts = 0;
            this->n_tris = 0;
            return;
        }

        this->ensure_used_cube_storage_size(this->n_used_cubes);
        
        // 7. Run your Compaction Kernel to fill 'used_cube_index' with the indices of active cubes
        compact_active_cubes_kernel<<<blocks, threads>>>(
            this->cube_codes, this->temp_buffer, n_cubes, this->used_cube_index, this->used_cube_code);
        CHECK_CUDA(cudaDeviceSynchronize());

        // 8. Run your Extraction Kernel to fill 'active_edges' with the edges of active cubes
        long long *d_all_edges;
        CHECK_CUDA(cudaMalloc(&d_all_edges, this->n_used_cubes * 12 * sizeof(long long)));

        // 9. Run the kernel to extract active edges
        int active_blocks = (this->n_used_cubes + threads - 1) / threads;
        extract_active_edges_kernel<<<active_blocks, threads>>>(
            cubes, 
            this->used_cube_index, 
            this->used_cube_code,
            d_all_edges, 
            this->n_used_cubes
        );
        CHECK_CUDA(cudaDeviceSynchronize());

        // 10. Sort and Unique
        thrust::device_ptr<long long> dev_all_edges(d_all_edges);
        thrust::sort(dev_all_edges, dev_all_edges + (this->n_used_cubes * 12));

        long long empty_edge = -1LL;
        auto valid_start = thrust::upper_bound(dev_all_edges, dev_all_edges + (this->n_used_cubes * 12), empty_edge);
        this->n_verts = thrust::distance(valid_start, thrust::unique(valid_start, dev_all_edges + (this->n_used_cubes * 12)));
        this->ensure_vert_storage_size(this->n_verts);

        // 11. Extract the unique edges to a separate array for interpolation
        CHECK_CUDA(cudaMemcpy(this->unique_edges, valid_start.get(), this->n_verts * sizeof(long long), cudaMemcpyDeviceToDevice));
        CHECK_CUDA(cudaFree(d_all_edges));

        // 12. Build Edge Map
        build_edge_map_kernel<<<active_blocks, threads>>>(
            cubes, 
            this->used_cube_index, 
            this->unique_edges, 
            this->cube_edge_to_vert_idx, 
            this->n_used_cubes, 
            this->n_verts
        );
        CHECK_CUDA(cudaDeviceSynchronize());

        // 13. Interpolate Vertices
        int vert_blocks = (this->n_verts + threads - 1) / threads;
        interpolate_vertices_kernel<<<vert_blocks, threads>>>(
            this->unique_edges,
            grid_vertices,
            values,
            this->n_verts,
            iso,
            this->verts
        );
        CHECK_CUDA(cudaDeviceSynchronize());

        // 14. Count the number of triangles for each active cube using the FACE_TABLE
        thrust::device_ptr<uint8_t> d_used_codes(this->used_cube_code);
        auto tri_count_iter = thrust::make_transform_iterator(d_used_codes, TriCountOp());

        thrust::device_ptr<IndexType> d_tri_prefix_sum(this->used_to_first_mc_tri);
        thrust::exclusive_scan(tri_count_iter, tri_count_iter + this->n_used_cubes, d_tri_prefix_sum);

        // 15. Get total number of triangles
        IndexType last_offset;
        uint8_t last_code;
        CHECK_CUDA(cudaMemcpy(&last_code, this->used_cube_code + this->n_used_cubes - 1, sizeof(uint8_t), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(&last_offset, this->used_to_first_mc_tri + this->n_used_cubes - 1, sizeof(IndexType), cudaMemcpyDeviceToHost));

        // int mc_values[2];
        // CHECK_CUDA(cudaMemcpyFromSymbol(mc_values, firstMarchingCubesId, 2 * sizeof(int), (last_code) * sizeof(int)));
        // int last_tri_len = mc_values[1] - mc_values[0];
        int last_tri_len =  h_firstMarchingCubesId[last_code + 1] - h_firstMarchingCubesId[last_code];
        this->n_tris = last_offset + last_tri_len; // Total indices (divide by 3 for triangle count)
        this->ensure_tri_storage_size(this->n_tris);

        // 16. Allocate output triangle array
        assemble_triangles_kernel<<<active_blocks, threads>>>(
            this->used_cube_code,
            this->used_to_first_mc_tri,
            this->cube_edge_to_vert_idx,
            this->n_used_cubes,
            this->tris
        );
        CHECK_CUDA(cudaDeviceSynchronize());
    };

    template <typename Scalar, typename IndexType>
    __global__ void backward_dmc_kernel(
        const long long* unique_edges,
        const Scalar* grid_values,
        const Vertex<Scalar>* grid_coords,
        const Vertex<Scalar>* adj_verts,
        IndexType n_verts,
        Scalar iso,
        Scalar* adj_values
    ) {
        int v_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (v_idx >= n_verts) return;

        // 1. Decode the unique edge to find the two grid vertex parents
        long long edge_sig = unique_edges[v_idx];
        IndexType v0_idx = static_cast<IndexType>(edge_sig >> 32);
        IndexType v1_idx = static_cast<IndexType>(edge_sig & 0xFFFFFFFF);

        // 2. Fetch the data needed for the chain rule
        Scalar v0_val = grid_values[v0_idx];
        Scalar v1_val = grid_values[v1_idx];
        Vertex<Scalar> p0 = grid_coords[v0_idx];
        Vertex<Scalar> p1 = grid_coords[v1_idx];

        Vertex<Scalar> grad_p_out = adj_verts[v_idx];

        // 3. Adjoint Math (Derivative of Linear Interpolation)
        Scalar diff = v1_val - v0_val;
        if (diff * diff < Scalar(1e-14)) return;

        // Project the 3D gradient onto the edge direction
        Scalar dot_prod = (p1 - p0).dot(grad_p_out);
        Scalar common = dot_prod / (diff * diff);

        // Calculate how the scalar values at the endpoints affect the vertex position
        Scalar grad_v0 = common * (iso - v1_val);
        Scalar grad_v1 = common * (v0_val - iso);

        // 4. Distribute the gradients back to the grid
        // Multiple unique edges share the same grid vertex, so we MUST use atomicAdd
        atomicAdd(&adj_values[v0_idx], grad_v0);
        atomicAdd(&adj_values[v1_idx], grad_v1);
    };

    template <typename Scalar, typename IndexType>
    void MC<Scalar, IndexType>::backward(
        Vertex<Scalar> const *grid_vertices,
        Scalar const *values,
        Vertex<Scalar> const *adj_verts, // Input Gradient (Mesh)
        Scalar *adj_values,              // Output Gradient (Grid Values)
        Scalar iso,
        int device
    ) {
        cudaSetDevice(device);
        // If no vertices were generated, there are no gradients to propagate
        if (this->n_verts == 0) return;
        int threads = 256;
        int blocks = (this->n_verts + threads - 1) / threads;
        backward_dmc_kernel<<<blocks, threads>>>(
            this->unique_edges,
            values,
            grid_vertices,
            adj_verts,
            this->n_verts,
            iso,
            adj_values
        );

        // Ensure the GPU finishes before returning to the framework
        CHECK_CUDA(cudaDeviceSynchronize());
    };

    // template struct MC<double, int>;
    template struct MC<float, int>;
    // template struct MC<__half, int>;

    // Explicit template instantiation for kernel functions

    // template __global__ void identify_active_cubes_kernel<double, int>(
    //     const int*, const double*, int, double, uint8_t*);
    template __global__ void identify_active_cubes_kernel<float, int>(
        const int*, const float*, int, float, uint8_t*);
    // template __global__ void identify_active_cubes_kernel<__half, int>(
    //     const int*, const __half*, int, __half, uint8_t*);

    template __global__ void compact_active_cubes_kernel<int>(
        const uint8_t*, const int*, int, int*, uint8_t*);

    template __global__ void extract_active_edges_kernel<int>(
        const int*, const int*, const uint8_t*, long long*, int);

    template __global__ void build_edge_map_kernel<int>(
        const int*, const int*, const long long*, int*, int, int);

    // template __global__ void interpolate_vertices_kernel<double, int>(
    //     const long long*, const Vertex<double>*, const double*, int, double, Vertex<double>*);
    template __global__ void interpolate_vertices_kernel<float, int>(
        const long long*, const Vertex<float>*, const float*, int, float, Vertex<float>*);
    // template __global__ void interpolate_vertices_kernel<__half, int>(
    //     const long long*, const Vertex<__half>*, const __half*, int, __half, Vertex<__half>*);

    template __global__ void assemble_triangles_kernel<int>(
        const uint8_t*, const int*, const int*, int, int*);

    template __global__ void backward_dmc_kernel<float, int>(
        const long long*, const float*, const Vertex<float>*, const Vertex<float>*, int, float, float*);
    // template __global__ void backward_dmc_kernel<__half, int>(
    //     const long long*, const __half*, const Vertex<__half>*, const Vertex<__half>*, int, __half, __half*);
}

template struct primitive::Vertex<float>;
// template struct primitive::Vertex<double>;
// template struct primitive::Vertex<__half>;
template struct primitive::Triangle<int>;