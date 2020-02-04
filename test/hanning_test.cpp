#include <gtest/gtest.h>

#include "hanning_host.hpp"

using namespace emida;

#define EXPECT_DOUBLE_VECTORS_EQ(x,y) \
	ASSERT_EQ(x.size(), y.size()) << "Vectors x and y are of unequal length"; \
	for (int i = 0; i < x.size(); ++i) \
	{ \
		EXPECT_DOUBLE_EQ(x[i], y[i]) << "Vectors " #x " and " #y " differ at index " << i; \
	}

void hann_data_load(std::string name, matrix<double> & a, matrix<double> & result)
{
	std::string test_location = "test/res/hann/";
	a = matrix<double>::from_file(test_location + name + "_A.txt");
	result = matrix<double>::from_file(test_location + name + "_res.txt");
}

TEST(hanning, hanning_vector)
{
	auto vec = hanning<double>(5);
	std::vector<double> expected = { 0, 0.5, 1, 0.5, 0 };
	EXPECT_DOUBLE_VECTORS_EQ(vec, expected);
}

TEST(hanning, hanning_window_5x5)
{
	matrix<double> a, expected;
	hann_data_load("5", a, expected);

	algorithm_hanning alg;

	alg.prepare(a);
	alg.run();
	alg.finalize();

	EXPECT_DOUBLE_VECTORS_EQ(alg.result().data, expected.data);
}

TEST(hanning, hanning_window_128x128)
{
	matrix<double> a, expected;
	hann_data_load("128", a, expected);

	algorithm_hanning alg;

	alg.prepare(a);
	alg.run();
	alg.finalize();

	EXPECT_DOUBLE_VECTORS_EQ(alg.result().data, expected.data);
}

