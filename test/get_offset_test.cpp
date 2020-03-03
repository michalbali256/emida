#include <gtest/gtest.h>

#include "get_offset.hpp"
#include "matrix.hpp"

using namespace emida;

template<typename T>
std::vector<T> get_submatrix(const T* src, vec2<size_t> src_size, vec2<size_t> begin, vec2<size_t> size)
{
	std::vector<T> res(size.x * size.y);
	for (size_t x = 0; x < size.x; ++x)
		for (size_t y = 0; y < size.y; ++y)
			res[y * size.x + x] = src[src_size.x * (y + begin.y) + x + begin.x];
	return res;
}

/*TEST(get_offset, small)
{
	vec2<size_t> src_size{ 10, 10 };
	vec2<size_t> size{ 5, 5 };
	std::vector<double> source(src_size.x * src_size.y);
	for (size_t i = 0; i < source.size(); ++i)
		source[i] = i;

	auto a = get_submatrix(source.data(), src_size, { 2, 2 }, size);
	auto b = get_submatrix(source.data(), src_size, { 4, 4 }, size);

	auto offset = get_offset<double>(b.data(), a.data(), size.x, size.y);

	EXPECT_DOUBLE_EQ(offset.x, 2);
	EXPECT_DOUBLE_EQ(offset.y, 2);

}*/

TEST(get_offset, bigger)
{
	matrix<double> pic = matrix<double>::from_file("test/res/data_pic.txt");
	matrix<double> temp = matrix<double>::from_file("test/res/data_temp.txt");

	vec2<size_t> src_size{ pic.n, pic.n };
	vec2<size_t> size{ 64, 64 };

	auto a = get_submatrix(pic.data.data(), src_size, { 0, 0 }, size);
	auto b = get_submatrix(temp.data.data(), src_size, { 0, 0 }, size);

	auto offset = get_offset<double>(a.data(), b.data(), size.x, size.y, 1);

	//results from test.py, first left topmost square
	//precision 1e-14 is OK, 1e-15 is failing
	EXPECT_NEAR(offset[0].x, 0.07583538046549165, 1e-14);
	EXPECT_NEAR(offset[0].y, -0.0982055210473689, 1e-14);

}
