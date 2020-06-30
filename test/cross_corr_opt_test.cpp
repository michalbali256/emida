#include <gtest/gtest.h>

#include "kernels.cuh"
#include "host_helpers.hpp"
#include "double_compare.hpp"

#include <cstdio>
using namespace emida;




TEST(cross_corr_opt, matrix_res5x5_block_5x5)
{
    std::vector<double> a =
    {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    };

    std::vector<double> b =
    {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    };

    std::vector<double> expected =
    {
        9, 26, 50, 38, 21,
        42, 94, 154, 106, 54,
        90, 186, 285, 186, 90,
        54, 106, 154, 94, 42,
        21, 38, 50, 26, 9
    };
    
    size2_t size = { 3, 3 };
    size2_t res_size = { 5, 5 };

    auto cu_a = vector_to_device(a);
    auto cu_b = vector_to_device(b);
    double* cu_res = cuda_malloc<double>(res_size.area());

    cuda_memset(cu_res, 0, res_size.area());

    run_cross_corr_opt(cu_a, cu_b, cu_res, size, res_size, {5,5}, 1, 1);

    CUCH(cudaGetLastError());
    CUCH(cudaDeviceSynchronize());

    auto res = device_to_vector(cu_res, res_size.area());

    EXPECT_DOUBLE_VECTORS_EQ(res, expected);
    

    int i = 0;
}

TEST(cross_corr_opt, matrix_res5x5_block_3x3)
{
    std::vector<double> a =
    {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    };

    std::vector<double> b =
    {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    };

    std::vector<double> expected =
    {
        9, 26, 50, 38, 21,
        42, 94, 154, 106, 54,
        90, 186, 285, 186, 90,
        54, 106, 154, 94, 42,
        21, 38, 50, 26, 9
    };

    size2_t size = { 3, 3 };
    size2_t res_size = { 5, 5 };

    auto cu_a = vector_to_device(a);
    auto cu_b = vector_to_device(b);
    double* cu_res = cuda_malloc<double>(res_size.area());

    cuda_memset(cu_res, 0, res_size.area());

    run_cross_corr_opt(cu_a, cu_b, cu_res, size, res_size, { 3,3 }, 1, 1);

    CUCH(cudaGetLastError());
    CUCH(cudaDeviceSynchronize());

    auto res = device_to_vector(cu_res, res_size.area());

    EXPECT_DOUBLE_VECTORS_EQ(res, expected);


    int i = 0;
}
