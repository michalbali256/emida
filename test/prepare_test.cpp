#include <gtest/gtest.h>

#include "matrix.hpp"
#include "kernels.cuh"
#include "double_compare.hpp"
#include "sums_serial.hpp"
#include "host_helpers.hpp"
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

std::vector<double> do_prepare(std::vector<uint16_t> picture,
	const std::vector<size2_t>& begins,
	const std::vector<double>& window_x,
	const std::vector<double>& window_y,
	const std::vector<double>& sums,
	size2_t pic_size,
	size2_t slice_size)
{
	double* cu_window_x = vector_to_device(window_x);
	double* cu_window_y = vector_to_device(window_y);
	uint16_t* cu_pic = vector_to_device(picture);
	double* cu_sums = vector_to_device(sums);
	size2_t* cu_begins = vector_to_device(begins);
	double* cu_slices = cuda_malloc<double>(begins.size() * slice_size.area());

	run_prepare_pics(cu_pic, cu_slices, cu_window_x, cu_window_y, cu_sums, cu_begins, pic_size, slice_size, begins.size());


	return device_to_vector(cu_slices, begins.size() * slice_size.area());
}

TEST(prepare, no_window)
{
	std::vector<uint16_t> picture =
	{
		1, 2, 3, 4, 5, 6,
		7, 8, 9, 10, 11, 12,
		13, 14, 15, 16, 17, 18,
		19, 20, 21, 22, 23, 24,
		25, 26, 27, 28, 29, 30
	};
	size2_t src_size = { 6,5 };
	std::vector<double> window_x(src_size.x, 1);
	std::vector<double> window_y(src_size.y, 1);
	size2_t slice_size = { 5, 3 };
	std::vector<size2_t> begins = { {1,2}, {0, 1} };
	std::vector<double> sums = sums_serial(picture, begins, src_size, slice_size);

	auto slices = do_prepare(picture, begins, window_x, window_y, sums, src_size, slice_size);

	std::vector<double> expected =
	{
		-8, -7, -6, -5, -4,
		-2, -1, 0, 1, 2,
		4, 5, 6, 7, 8,

		-8, -7, -6, -5, -4,
		-2, -1, 0, 1, 2,
		4, 5, 6, 7, 8
	};

	EXPECT_DOUBLE_VECTORS_EQ(slices, expected);
}

TEST(prepare, window)
{
	std::vector<uint16_t> picture =
	{
		1, 2, 3, 4, 5, 6,
		7, 8, 9, 10, 11, 12,
		13, 14, 15, 16, 17, 18,
		19, 20, 21, 22, 23, 24,
		25, 26, 27, 28, 29, 30
	};
	size2_t src_size = { 6,5 };
	std::vector<double> window_x(src_size.x, 1);
	window_x[2] = 0.5;
	std::vector<double> window_y(src_size.y, 1);
	window_y[1] = 0.25;
	size2_t slice_size = { 5, 3 };
	std::vector<size2_t> begins = { {1,2}, {0, 1} };
	std::vector<double> sums = sums_serial(picture, begins, src_size, slice_size);

	auto slices = do_prepare(picture, begins, window_x, window_y, sums, src_size, slice_size);

	std::vector<double> expected =
	{
		-8, -7, -3, -5, -4,
		-0.5, -.25, 0, .25, .5,
		4, 5, 3, 7, 8,

		-8, -7, -3, -5, -4,
		-.5, -.25, 0, .25, .5,
		4, 5, 3, 7, 8
	};

	EXPECT_DOUBLE_VECTORS_EQ(slices, expected);
}
/*
TEST(hanning, hanning_window_5x5)
{
	auto hann_window_x = generate_hanning<T>(cols);
	auto hann_window_y = generate_hanning<T>(rows);
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
*/
