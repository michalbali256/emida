#include <gtest/gtest.h>

#include "cross_corr_host.hpp"

#include "tiffio.h"
#include <cstdio>
using namespace emida;

void cross_corr_data_load(std::string name, matrix<int> & a, matrix<int> & b, matrix<int> & result)
{
    std::string test_location = "test/res/cross_corr/";
    a = matrix<int>::from_file(test_location + name + "_A.txt");
    b = matrix<int>::from_file(test_location + name + "_B.txt");
    result = matrix<int>::from_file(test_location + name + "_res.txt");
}

TEST(cross_corr, matrix_3x3)
{
    algorithm_cross_corr<int> alg;

    std::vector<int> a =
    {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    };

    std::vector<int> expected =
    {
        9, 26, 50, 38, 21,
        42, 94, 154, 106, 54,
        90, 186, 285, 186, 90,
        54, 106, 154, 94, 42,
        21, 38, 50, 26, 9
    };


    alg.prepare(a.data(), a.data(), { 3, 3 }, {5, 5}, 1);
    alg.run();
    alg.finalize();

    EXPECT_EQ(expected, alg.result());
}

TEST(cross_corr, matrix_3x3_res3x3)
{
    algorithm_cross_corr<int> alg;

    std::vector<int> a =
    {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    };

    std::vector<int> expected =
    {
      94, 154, 106,
      186, 285, 186,
      106, 154, 94
    };


    alg.prepare(a.data(), a.data(), { 3, 3 }, { 3, 3 }, 1);
    alg.run();
    alg.finalize();

    EXPECT_EQ(expected, alg.result());
}

TEST(cross_corr, matrix_4x3)
{
    algorithm_cross_corr<int> alg;

    std::vector<int> a =
    {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12
    };

    std::vector<int> b =
    {
        8, 7, 6, 5,
        16, 15, 14, 13,
        4, 3, 2, 1
    };

    std::vector<int> res =
    {
        1, 4, 10, 20, 25, 24, 16,
        18, 56, 116, 200, 194, 160, 96,
        79, 192, 342, 532, 471, 364, 208,
        142, 316, 524, 768, 638, 468, 256,
        45, 104, 178, 268, 229, 172, 96
    };


    alg.prepare(a.data(), b.data(), { 4, 3 }, {7, 5}, 1);
    alg.run();
    alg.finalize();

    EXPECT_EQ(res, alg.result());
}

TEST(cross_corr, matrix_4x3x2)
{
    algorithm_cross_corr<int> alg;

    std::vector<int> a =
    {
        1, 2, 3, 4 ,
        5, 6, 7, 8,
        9, 10, 11, 12,

        3, 2, 1, 4,
        2, 6, 10, 8,
        9, 8, 11, 5
    };

    std::vector<int> b =
    {
        8, 7, 6, 5,
        16, 15, 14, 13,
        4, 3, 2, 1,

        1, 2, 3, 4,
        16, 5, 14, 13,
        8, 3, 2, 1
    };

    std::vector<int> res =
    {
        1, 4, 10, 20, 25, 24, 16,
        18, 56, 116, 200, 194, 160, 96,
        79, 192, 342, 532, 471, 364, 208,
        142, 316, 524, 768, 638, 468, 256,
        45, 104, 178, 268, 229, 172, 96,

        3, 8, 14, 36, 27, 20, 32,
        41, 78, 84, 186, 187, 140, 128,
        47, 149, 294, 455, 381, 312, 172,
        125, 260, 362, 479, 303, 227, 88,
        36, 59, 86, 78, 45, 21, 5
    };


    alg.prepare(a.data(), b.data(), { 4, 3 }, {7, 5}, 2);
    alg.run();
    alg.finalize();

    EXPECT_EQ(res, alg.result());
}

TEST(cross_corr, matrix_4x3x2_res5x5)
{
    algorithm_cross_corr<int> alg;

    std::vector<int> a =
    {
        1, 2, 3, 4 ,
        5, 6, 7, 8,
        9, 10, 11, 12,

        3, 2, 1, 4,
        2, 6, 10, 8,
        9, 8, 11, 5
    };

    std::vector<int> b =
    {
        8, 7, 6, 5,
        16, 15, 14, 13,
        4, 3, 2, 1,

        1, 2, 3, 4,
        16, 5, 14, 13,
        8, 3, 2, 1
    };

    std::vector<int> res =
    {
        4, 10, 20, 25, 24,
        56, 116, 200, 194, 160,
        192, 342, 532, 471, 364,
        316, 524, 768, 638, 468,
        104, 178, 268, 229, 172,

        8, 14, 36, 27, 20,
        78, 84, 186, 187, 140,
        149, 294, 455, 381, 312,
        260, 362, 479, 303, 227,
        59, 86, 78, 45, 21
    };


    alg.prepare(a.data(), b.data(), { 4, 3 }, { 5, 5 }, 2);
    alg.run();
    alg.finalize();

    EXPECT_EQ(res, alg.result());
}


TEST(cross_corr, matrix_64x64)
{
    algorithm_cross_corr<int> alg;
    matrix<int> a, b, res;
    cross_corr_data_load("64", a, b, res);

    vec2<size_t> size = { a.n, a.n };

    alg.prepare(a.data.data(), b.data.data(), size , (size*2) - 1, 1);
    alg.run();
    alg.finalize();

    EXPECT_EQ(res.data, alg.result());
}