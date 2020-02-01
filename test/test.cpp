#include <gtest/gtest.h>

#include "cross_corr_host.hpp"

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
	algorithm_cross_corr alg;
	matrix<int> a, b, res;
	cross_corr_data_load("3", a, b, res);

	alg.prepare(a, b);
	alg.run();
	alg.finalize();

	EXPECT_EQ(res, alg.result());
}

TEST(cross_corr, matrix_64x64)
{
	algorithm_cross_corr alg;
	matrix<int> a, b, res;
	cross_corr_data_load("64", a, b, res);

	alg.prepare(a, b);
	alg.run();
	alg.finalize();

	EXPECT_EQ(res, alg.result());
}