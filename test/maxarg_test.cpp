#include <gtest/gtest.h>

#include "maxarg_host.hpp"

using namespace emida;

TEST(extract_neighbors, size_8)
{
	std::vector<double> v(25);
	for (size_t i = 0; i < v.size(); ++i)
		v[i] = i;
	double * cu_data = vector_to_device(v);
	double* cu_neigh;
	cudaMalloc(&cu_neigh, 9 * sizeof(double));
	run_extract_neighbors<double, 3>(cu_data, cu_neigh, 3, 3, 5, 5);

	std::vector<double> neigh(9);
	cudaMemcpy(neigh.data(), cu_neigh, 9 * sizeof(double), cudaMemcpyDeviceToHost);

	std::vector<double> expected = {12, 13, 14, 17, 18, 19, 22, 23, 24};

	EXPECT_EQ(neigh, expected);
}

TEST(maxarg, size_8)
{
	std::vector<double> v = {1,3, 5, 8, 10, 4, 3, 2};
	algorithm_maxarg<double> a;
	a.prepare(v);
	a.run();
	a.finalize();

	EXPECT_EQ(a.result(), 4U);
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
	a.prepare(data);
	a.run();
	a.finalize();

	EXPECT_EQ(a.result(), 1001);
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
	a.prepare(data);
	a.run();
	a.finalize();

	EXPECT_EQ(a.result(), 1024);
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
	a.prepare(data);
	a.run();
	a.finalize();

	EXPECT_EQ(a.result(), 100000);
}