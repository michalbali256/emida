#include <gtest/gtest.h>

#include "maxarg_host.hpp"

using namespace emida;


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