#include <gtest/gtest.h>

#include "slice_picture.hpp"

using namespace emida;

TEST(slice_picture, small)
{
	std::vector<int> v =
	{
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
		13, 14, 15, 16
	};

	std::vector<int> expected =
	{
		1, 2,
		5, 6,

		3, 4,
		7, 8,

		9, 10,
		13, 14,

		11, 12,
		15, 16
	};

	auto sliced = slice_picture(v.data(), { 4, 4 }, { 2, 2 }, { 2, 2 });

	EXPECT_EQ(sliced, expected);

}