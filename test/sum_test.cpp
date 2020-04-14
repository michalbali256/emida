#include <gtest/gtest.h>

#include <tuple>

#include "maxarg_host.hpp"
#include "get_offset.hpp"
#include "double_compare.hpp"

using namespace emida;

std::vector<double> do_sums(const std::vector<double> & data, size_t size, size_t batch_size)
{
	auto cu_data = vector_to_device(data);
	auto cu_sums = cuda_malloc<double>(batch_size);

	run_sum(cu_data, cu_sums, size, batch_size);

	CUCH(cudaGetLastError());
	CUCH(cudaDeviceSynchronize());

	return device_to_vector(cu_sums, batch_size);
}

std::vector<double> do_sums_slice(const std::vector<double>& data, const std::vector<size2_t>& begins, size2_t src_size, size2_t slice_size)
{
	size_t batch_size = begins.size();
	auto cu_data = vector_to_device(data);
	auto cu_begins = vector_to_device(begins);
	auto cu_sums = cuda_malloc<double>(batch_size);

	run_sum(cu_data, cu_sums, cu_begins, src_size, slice_size, batch_size);

	CUCH(cudaGetLastError());
	CUCH(cudaDeviceSynchronize());

	return device_to_vector(cu_sums, batch_size);
}

TEST(sum, size_8)
{
	std::vector<double> v = { 1, 3, 5, 8, 10, 4, 3, 2};
	
	auto sums = do_sums(v, 8, 1);

	EXPECT_EQ(sums[0], 36);
}

TEST(sum_slice, size_9)
{
	std::vector<double> v =
	{
		1, 3, 5, 8,
		8, 10, 4, 1,
		3, 2, 5, 15
	};

	auto sums = do_sums_slice(v, { {0, 0}, {1, 1} }, { 4,3 }, { 2,2 });

	EXPECT_EQ(sums[0], 22);
	EXPECT_EQ(sums[1], 21);
}

TEST(sum, size_8x2)
{
	std::vector<double> v =
	{
		1,  3,  5,  8, 10, 4, 3, 2,
		1, 12, 15, 18, 10, 4, 3, 2
	};

	auto sums = do_sums(v, 8, 2);

	EXPECT_EQ(sums[0], 36);
	EXPECT_EQ(sums[1], 65);
}


TEST(sums, size_1333x3)
{
	std::vector<double> data;
	data.resize(3999);
	double mom = 0;
	for (size_t i = 0; i < data.size(); ++i)
		data[i] = (double)i;
	
	auto sums = do_sums(data, 1333, 3);

	ASSERT_EQ(sums.size(), 3);
	EXPECT_EQ(sums[0], 1332 * 1333 / 2);
	EXPECT_EQ(sums[1], (1333 + 2665) * 1333 / 2);
	EXPECT_EQ(sums[2], (2666 + 3998) * 1333 / 2);
}

class sums_slice_fixture :public ::testing::TestWithParam<std::tuple<size2_t, size2_t, std::vector<size2_t>>> {

};

INSTANTIATE_TEST_SUITE_P(
	sums_slicee,
	sums_slice_fixture,
	::testing::Values(
		std::make_tuple < size2_t, size2_t, std::vector<size2_t>>({ 31,43 }, { 29,40 }, { { 0, 0 }, { 1, 1 } }),
		std::make_tuple < size2_t, size2_t, std::vector<size2_t>>({ 64,64 }, { 32,32 }, { { 0, 0 }, { 0, 16 }, { 16, 16 }, { 32, 32 } })
	)
);


TEST_P(sums_slice_fixture, size_)
{
	size2_t src_size = std::get<0>(GetParam());
	size2_t slice_size = std::get<1>(GetParam());
	std::vector<double> data;
	data.resize(src_size.area());
	double mom = 0;
	for (size_t i = 0; i < data.size(); ++i)
		data[i] = (double)i;

	std::vector<size2_t> begins = std::get<2>(GetParam());

	auto sums = do_sums_slice(data, begins, src_size, slice_size);

	std::vector<double> expected_sums(begins.size());
	for (size_t i = 0; i < begins.size(); ++i)
	{
		for (size_t x = 0; x < slice_size.x; ++x)
			for (size_t y = 0; y < slice_size.y; ++y)
			{
				size2_t from = begins[i] + size2_t{ x, y };
				expected_sums[i] += data[from.pos(src_size.x)];
			}
	}

	ASSERT_EQ(sums.size(), begins.size());
	
}

TEST(sums, size_4096x2)
{
	std::vector<double> data;
	data.resize(8192);
	double mom = 0;
	for (size_t i = 0; i < data.size(); ++i)
		data[i] = (double)i;

	auto sums = do_sums(data, 4096, 2);

	ASSERT_EQ(sums.size(), 2);
	EXPECT_EQ(sums[0], 4095 * 4096 / 2);
	EXPECT_EQ(sums[1], (4096 + 8191) * 4096 / 2);
}
