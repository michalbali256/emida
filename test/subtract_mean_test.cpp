#include <vector>

#include "gtest/gtest.h"

#include "subtract_mean.hpp"
#include "double_compare.hpp"

TEST(subtract_mean, simple)
{
	std::vector<double> v{ 1, 5, 8, 1, 2, 9, 4, 2, 5, 3 };
	std::vector<double> expected{ -3, 1, 4, -3, -2, 5, 0, -2, 1, -1 };
	
	emida::subtract_mean(v.data(), v.size(), 1);
	
	EXPECT_DOUBLE_VECTORS_EQ(v, expected);
	
}

TEST(subtract_mean, small_batches)
{
	std::vector<double> v =
	{
		1, 5,
		8, 1,
		2, 10,
		4, 2,
		5, 3
	};
	std::vector<double> expected =
	{
		-2, 2,
		3.5, -3.5,
		-4, 4,
		1, -1,
		1, -1
	};

	emida::subtract_mean(v.data(), 2, 5);

	EXPECT_DOUBLE_VECTORS_EQ(v, expected);

}