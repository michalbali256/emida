#include <gtest/gtest.h>

#include "maxarg_host.hpp"
#include "get_offset.hpp"

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

TEST(sum, size_8)
{
	std::vector<double> v = { 1, 3, 5, 8, 10, 4, 3, 2};
	
	auto sums = do_sums(v, 8, 1);

	EXPECT_EQ(sums[0], 36);
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
		data[i] = i;
	
	auto sums = do_sums(data, 1333, 3);

	ASSERT_EQ(sums.size(), 3);
	EXPECT_EQ(sums[0], 1332 * 1333 / 2);
	EXPECT_EQ(sums[1], (1333 + 2665) * 1333 / 2);
	EXPECT_EQ(sums[2], (2666 + 3998) * 1333 / 2);
}

TEST(sums, size_4096x2)
{
	std::vector<double> data;
	data.resize(8192);
	double mom = 0;
	for (size_t i = 0; i < data.size(); ++i)
		data[i] = i;

	auto sums = do_sums(data, 4096, 2);

	ASSERT_EQ(sums.size(), 2);
	EXPECT_EQ(sums[0], 4095 * 4096 / 2);
	EXPECT_EQ(sums[1], (4096 + 8191) * 4096 / 2);
}
