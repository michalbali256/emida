#include <gtest/gtest.h>

#include "hanning_host.hpp"
#include "double_compare.hpp"
using namespace emida;


void hann_data_load(std::string name, matrix<double> & a, matrix<double> & result)
{
	std::string test_location = "test/res/hann/";
	a = matrix<double>::from_file(test_location + name + "_A.txt");
	result = matrix<double>::from_file(test_location + name + "_res.txt");
}

TEST(hanning, hanning_vector)
{
	auto vec = generate_hanning<double>(5);
	std::vector<double> expected = { 0, 0.5, 1, 0.5, 0 };
	EXPECT_DOUBLE_VECTORS_EQ(vec, expected);
}

TEST(hanning, hanning_window_5x5)
{
	matrix<double> a, expected;
	hann_data_load("5", a, expected);

	algorithm_hanning<double> alg;

	alg.prepare(a.data.data(), 5, 5, 1);
	alg.run();
	alg.finalize();

	EXPECT_DOUBLE_VECTORS_EQ(alg.result(), expected.data);
}

TEST(hanning, hanning_window_128x128)
{
	matrix<double> a, expected;
	hann_data_load("128", a, expected);

	algorithm_hanning<double> alg;

	alg.prepare(a.data.data(), 128, 128, 1);
	alg.run();
	alg.finalize();

	EXPECT_DOUBLE_VECTORS_EQ(alg.result(), expected.data);
}

TEST(hanning, hanning_window_3x3x2)
{
	std::vector<double> a =
	{
		1, 1, 1, 1, 1,
		1, 1, 1, 1, 1,
		1, 1, 1, 1, 1,
		2, 2, 2, 2, 2,
		2, 2, 2, 2, 2,
		2, 2, 2, 2, 2
	};
	std::vector<double> expected = 
	{
		0, 0, 0, 0, 0,
		0, 0.5, 1, 0.5, 0,
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,
		0, 1, 2, 1, 0,
		0, 0, 0, 0, 0
	};

	algorithm_hanning<double> alg;

	alg.prepare(a.data(), 5, 3, 2);
	alg.run();
	alg.finalize();

	EXPECT_DOUBLE_VECTORS_EQ(alg.result(), expected);
}

TEST(hanning, hanning_window_128x128x5)
{
	matrix<double> a, expected;
	hann_data_load("128", a, expected);

	

	size_t repeat = 5;
	auto pics = repeat_vector(a.data, repeat);
	auto expected_pics = repeat_vector(expected.data, repeat);

	algorithm_hanning<double> alg;

	alg.prepare(pics.data(), 128, 128, repeat);
	alg.run();
	alg.finalize();

	EXPECT_DOUBLE_VECTORS_EQ(alg.result(), expected_pics);
}