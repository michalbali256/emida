#include <gtest/gtest.h>

#include "subpixel_max.hpp"
#include "double_compare.hpp"

using namespace emida;

TEST(lstsq_matrix, matrix)
{
	auto m = get_matrix<3>();

	std::array<double, 3 * 3 * 6> e = {
		1, 0, 0, 0, 0, 0,
		1, 1, 0, 1, 0, 0,
		1, 2, 0, 4, 0, 0,
		1, 0, 1, 0, 0, 1,
		1, 1, 1, 1, 1, 1,
		1, 2, 1, 4, 2, 1,
		1, 0, 2, 0, 0, 4,
		1, 1, 2, 1, 2, 4,
		1, 2, 2, 4, 4, 4,
	};

	EXPECT_EQ(m, e);
}

TEST(multiply, small)
{
	std::vector<int> a = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
	std::vector<int> b = { 4, 3, 2, 1, 3, 3, 4, 1 };
	std::vector<int> expected = { 33, 18, 85, 50, 137, 82 };

	std::vector<int> c(6);

	multiply(a.data(), b.data(), c.data(), 3, 2, 4);

	EXPECT_EQ(c, expected);
}

TEST(subpixel_max_serial, coefs)
{
	std::array<double, 9> pic = {1, 2, 3, 4, 5, 6, 7, 8, 9};
	auto coef = subpixel_max_coefs<double>(pic.data(), 3);
	std::array<double, 6> expected = {1, 1, 3, 0, 0, 0};

	EXPECT_DOUBLE_VECTORS_NEAR(coef, expected, 1e-15)
}

TEST(subpixel_max_serial, coefs2)
{
	std::array<double, 9> pic = { 1, 2, 3, 7, 9, 8, 4, 5, 6 };
	auto coef = subpixel_max_coefs<double>(pic.data(), 3);
	std::array<double, 6> expected = { 1.0, 1.833333333333333, 10.5, -0.5, 0, -4.5 };

	EXPECT_DOUBLE_VECTORS_EQ(coef, expected)
}

TEST(subpixel_max_serial, small)
{
	std::array<double, 9> pic = { 1, 2, 3, 7, 9, 8, 4, 5, 6 };
	auto [d, _] = subpixel_max_serial<double>(pic.data(), 3, 1);
	std::vector<vec2<double>> expected = { { 1.8333333333333333, 1.1666666666666666 } };
	
	EXPECT_VEC_VECTORS_EQ(d, expected);
}
