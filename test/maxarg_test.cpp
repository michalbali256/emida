#include <gtest/gtest.h>

#include "maxarg_host.hpp"
#include "get_offset.hpp"

using namespace emida;

TEST(extract_neighbors, size_5x5)
{
	std::vector<double> v(25);
	for (size_t i = 0; i < v.size(); ++i)
		v[i] = (double)i;
	double * cu_data = vector_to_device(v);
	
	double* cu_neigh;
	cudaMalloc(&cu_neigh, 9 * sizeof(double));

	std::vector<vec2<size_t>> max_i = { {3,3} };
	auto cu_max_i = vector_to_device(max_i);

	run_extract_neighbors<double, 3>(cu_data, cu_max_i, cu_neigh, 5, 5, 1);

	std::vector<double> neigh(9);
	cudaMemcpy(neigh.data(), cu_neigh, 9 * sizeof(double), cudaMemcpyDeviceToHost);

	std::vector<double> expected = {12, 13, 14, 17, 18, 19, 22, 23, 24};

	EXPECT_EQ(neigh, expected);
}

TEST(extract_neighbors, size_5x5x2)
{
	std::vector<double> v(50);
	for (size_t i = 0; i < v.size(); ++i)
		v[i] = (double) i;
	double* cu_data = vector_to_device(v);

	double* cu_neigh;
	size_t neigh_size = 3 * 3 * 2;
	cudaMalloc(&cu_neigh, neigh_size * sizeof(double));

	std::vector<vec2<size_t>> max_i = { {3,3}, {1,1} };
	auto cu_max_i = vector_to_device(max_i);

	run_extract_neighbors<double, 3>(cu_data, cu_max_i, cu_neigh, 5, 5, 2);

	std::vector<double> neigh = device_to_vector(cu_neigh, neigh_size);

	std::vector<double> expected =
	{
		12, 13, 14,
		17, 18, 19,
		22, 23, 24,
	
		25, 26, 27,
		30, 31, 32,
		35, 36, 37
	};

	EXPECT_EQ(neigh, expected);
}

TEST(maxarg, size_8)
{
	std::vector<double> v = {1,3, 5, 8, 10, 4, 3, 2};
	algorithm_maxarg<double> a;
	a.prepare(v, 8, 1);
	a.run();
	a.finalize();

	EXPECT_EQ(a.result()[0], 4U);
}

TEST(maxarg, size_8x2)
{
	std::vector<double> v =
	{
		1,  3,  5,  8, 10, 4, 3, 2,
		1, 12, 15, 18, 10, 4, 3, 2
	};
	algorithm_maxarg<double> a;
	a.prepare(v, 8, 2);
	a.run();
	a.finalize();

	EXPECT_EQ(a.result()[0], 4U);
	EXPECT_EQ(a.result()[1], 11U);
}

TEST(maxarg, size_1333)
{
	std::vector<double> data;
	data.resize(1333);
	double mom = 0;
	for (auto& val : data)
	{
		val = mom;
		mom += 0.007;
		if (mom > 1)
			mom = -2;
	}
	data[1001] = 3;

	algorithm_maxarg<double> a;
	a.prepare(data, 1333, 1);
	a.run();
	a.finalize();

	EXPECT_EQ(a.result()[0], 1001);
}

TEST(maxarg, size_1333x3)
{
	std::vector<double> data;
	data.resize(3999);
	double mom = 0;
	for (auto& val : data)
	{
		val = mom;
		mom += 0.007;
		if (mom > 1)
			mom = -2;
	}
	data[1001] = 3;
	data[1024] = 4;
	data[1333] = 4;
	data[3000] = 4;
	algorithm_maxarg<double> a;
	a.prepare(data, 1333, 3);
	a.run();
	a.finalize();

	ASSERT_EQ(a.result().size(), 3);
	EXPECT_EQ(a.result()[0], 1024);
	EXPECT_EQ(a.result()[1], 1333);
	EXPECT_EQ(a.result()[2], 3000);
}

TEST(maxarg, size_4096)
{
	std::vector<double> data;
	data.resize(4096);
	double mom = 0;
	for (auto& val : data)
	{
		val = mom;
		mom += 0.007;
		if (mom > 1)
			mom = -2;
	}
	data[1024] = 3;

	algorithm_maxarg<double> a;
	a.prepare(data, 4096, 1);
	a.run();
	a.finalize();

	EXPECT_EQ(a.result()[0], 1024);
}

TEST(maxarg, size_100100)
{
	std::vector<double> data;
	data.resize(100100);
	double mom = 0;
	for (auto& val : data)
	{
		val = mom;
		mom += 0.007;
		if (mom > 1)
			mom = -2;
	}
	data[100000] = 3;

	algorithm_maxarg<double> a;
	a.prepare(data, 100100, 1);
	a.run();
	a.finalize();

	EXPECT_EQ(a.result()[0], 100000);
}


TEST(maxarg, size_4x3x2)
{
	std::vector<double> v =
	{
		1,  3,  5,  8,
		10, 4, 3, 2,
		1, 2, 3, 4,

		1, 12, 15, 18,
		10, 4, 3, 2,
		1, 2, 3, 4
	};

	double* cu_data = vector_to_device(v);

	auto res = get_maxarg(cu_data, { 4, 3 }, 2);

	EXPECT_EQ(res[0].x, 0U);
	EXPECT_EQ(res[0].y, 1U);
	EXPECT_EQ(res[1].x, 3U);
	EXPECT_EQ(res[1].y, 0U);

}