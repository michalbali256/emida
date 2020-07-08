#include <gtest/gtest.h>

#include "kernels.cuh"
#include "host_helpers.hpp"
#include "double_compare.hpp"
#include "stringer.hpp"
#include "cross_corr.hpp"

#include <cstdio>
using namespace emida;

TEST(cross_corr_serial, matrix_4x3)
{
    std::vector<double> a =
    {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12
    };

    std::vector<double> b =
    {
        8, 7, 6, 5,
        16, 15, 14, 13,
        4, 3, 2, 1
    };

    std::vector<double> expected =
    {
        1, 4, 10, 20, 25, 24, 16,
        18, 56, 116, 200, 194, 160, 96,
        79, 192, 342, 532, 471, 364, 208,
        142, 316, 524, 768, 638, 468, 256,
        45, 104, 178, 268, 229, 172, 96
    };

    std::vector<double> res = cross_corr_serial(a.data(), b.data(), {4,3}, 1);

    EXPECT_DOUBLE_VECTORS_EQ(res, expected);
}

TEST(cross_corr_serial, matrix_4x3x2)
{
    std::vector<double> a =
    {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,

        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12
    };

    std::vector<double> b =
    {
        8, 7, 6, 5,
        16, 15, 14, 13,
        4, 3, 2, 1,

        8, 7, 6, 5,
        16, 15, 14, 13,
        4, 3, 2, 1,
    };

    std::vector<double> expected =
    {
        1, 4, 10, 20, 25, 24, 16,
        18, 56, 116, 200, 194, 160, 96,
        79, 192, 342, 532, 471, 364, 208,
        142, 316, 524, 768, 638, 468, 256,
        45, 104, 178, 268, 229, 172, 96,

        1, 4, 10, 20, 25, 24, 16,
        18, 56, 116, 200, 194, 160, 96,
        79, 192, 342, 532, 471, 364, 208,
        142, 316, 524, 768, 638, 468, 256,
        45, 104, 178, 268, 229, 172, 96
    };

    std::vector<double> res = cross_corr_serial(a.data(), b.data(), { 4,3 }, 2);

    EXPECT_DOUBLE_VECTORS_EQ(res, expected);
}


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

    run_cross_corr_r(cu_a, cu_b, cu_res, size, res_size, 1, 1);

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

struct corr_param
{
    size2_t size;
    size2_t res_size;
    size2_t block_size;
    size_t slices;
    std::string name;
};

class cross_corr_test : public ::testing::TestWithParam<corr_param> {

};

/*INSTANTIATE_TEST_SUITE_P(
    cross_corr_opt,
    cross_corr_test,
    ::testing::Values(
        corr_param{ {3, 3}, {5, 5}, {3, 3}, 1, "matrix_res5x5_block3x3" },
        corr_param{ {3, 3}, {5, 5}, {3, 3}, 2, "matrix_res5x5_block3x3_slicesx2" },
        corr_param{ {3, 3}, {5, 5}, {5, 5}, 1, "matrix_res5x5_block5x5" }
        corr_param{ {3, 3}, {5, 5}, {5, 5}, 2, "matrix_res5x5_block5x5_slicesx2" },
        corr_param{ {5, 5}, {9, 9}, {5, 5}, 1, "matrix_res9x9_block5x5" },
        corr_param{ {5, 5}, {9, 9}, {5, 5}, 2, "matrix_res9x9_block5x5_slicesx2" },
        corr_param{ {5, 5}, {9, 9}, {9, 9}, 1, "matrix_res9x9_block9x9" },
        corr_param{ {5, 5}, {9, 9}, {9, 9}, 2, "matrix_res9x9_block9x9_slicesx2" },
        corr_param{ {96, 96}, {191, 191}, {31, 31}, 1, "matrix_res191x191_block31x31" },
        corr_param{ {96, 96}, {191, 191}, {31, 31}, 2, "matrix_res191x191_block31x31_slicesx2" }
    ),
    stringer<corr_param>()
);*/

/*INSTANTIATE_TEST_SUITE_P(
    cross_corr_opt,
    cross_corr_test,
    ::testing::Values(
        corr_param{ {5, 5}, {9, 9}, {5, 5}, 1, "matrix_res9x9_block5x5" },
        corr_param{ {96, 96}, {191, 191}, {31, 31}, 1, "matrix_res191x191_block31x31" },
        corr_param{ {96, 96}, {191, 191}, {31, 31}, 2, "matrix_res191x191_block31x31_slx2" }
    ),
    stringer<corr_param>()
);*/

TEST_P(cross_corr_test, size_)
{
    double val = 1;
    std::vector<double> a(GetParam().size.area() * GetParam().slices);
    for (size_t i = 0; i < a.size(); ++i)
        a[i] = val++;
    
    std::vector<double> b(GetParam().size.area() * GetParam().slices);
    for (size_t i = 0; i < b.size(); ++i)
        b[i] = val++;

    std::vector<double> expected = cross_corr_serial(a.data(), b.data(), GetParam().size, GetParam().slices);

    auto cu_a = vector_to_device(a);
    auto cu_b = vector_to_device(b);
    double* cu_res = cuda_malloc<double>(GetParam().res_size.area() * GetParam().slices);

    cuda_memset(cu_res, 0, GetParam().res_size.area());
    
    auto b_s = GetParam().block_size;
    run_cross_corr_r(cu_a, cu_b, cu_res, GetParam().size, GetParam().res_size, GetParam().slices, 1);

    CUCH(cudaGetLastError());
    CUCH(cudaDeviceSynchronize());

    auto res = device_to_vector(cu_res, GetParam().res_size.area() * GetParam().slices);

    

    EXPECT_DOUBLE_VECTORS_EQ(res, expected);

    int i = 0;
}
