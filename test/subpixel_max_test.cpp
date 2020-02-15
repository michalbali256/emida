#include <gtest/gtest.h>

#include "subpixel_max.hpp"
#include "double_compare.hpp"


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

TEST(lstsq_matrix, subpixel_max)
{
	std::array<double, 9> pic = {1, 2, 3, 4, 5, 6, 7, 8, 9};
	auto coef = subpixel_max_coefs<double, 3>(pic.data());
	std::array<double, 6> expected = {1, 1, 3, 0, 0, 0};

	EXPECT_DOUBLE_VECTORS_NEAR(coef, expected, 1e-15)

}
