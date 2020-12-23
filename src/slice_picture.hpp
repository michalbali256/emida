#pragma once

#include <cassert>

#include "common.hpp"

namespace emida
{

template<typename T, typename RES>
inline void copy_submatrix(const T* __restrict__ src, RES* __restrict__ dst, size2_t src_size, size2_t begin, size2_t size)
{
	for (esize_t y = 0; y < size.y; ++y)
		memcpy(dst + y * size.x, src + src_size.x * (y + begin.y) + begin.x, size.x * sizeof(T));
		//for (esize_t x = 0; x < size.x; ++x)
		//	dst[y * size.x + x] = src[src_size.x * (y + begin.y) + x + begin.x];
}

template<typename T, typename RES>
inline std::vector<RES> get_submatrix(const T* src, size2_t src_size, size2_t begin, size2_t size)
{
	std::vector<RES> res(size.x * size.y);
	copy_submatrix(src, res.data(), src_size, begin, size);
	return res;
}

inline esize_t get_sliced_batch_size(size2_t src_size, size2_t size, size2_t step)
{
	return ((src_size.x - size.x) / step.x + 1) *
		((src_size.y - size.y) / step.y + 1);
}

template<typename T>
inline std::vector<T> slice_picture(const T* src, size2_t src_size, size2_t size, size2_t step)
{
	assert(src_size.x % size.x == 0);
	assert(src_size.y % size.y == 0);

	std::vector<T> res(get_sliced_batch_size(src_size, size, step) * size.x * size.y);

	T* next = res.data();
	size2_t i = { 0,0 };
	for (i.y = 0; i.y + size.y <= src_size.y; i.y += step.y)
		for (i.x = 0; i.x + size.x <= src_size.x; i.x += step.x)
		{
			copy_submatrix(src, next, src_size, i, size);

			next += size.x * size.y;
		}

	return res;
}

inline std::vector<size2_t> get_slice_begins(size2_t src_size, size2_t size, size2_t step)
{
	std::vector<size2_t> res;
	res.reserve(get_sliced_batch_size(src_size, size, step));

	size2_t i = { 0,0 };
	for (i.y = 0; i.y + size.y <= src_size.y; i.y += step.y)
		for (i.x = 0; i.x + size.x <= src_size.x; i.x += step.x)
			res.push_back(i);
	return res;
}

template< typename RES, typename IN>
inline std::vector<RES> get_pics(const IN* src, size2_t src_size, const std::vector<size2_t>& begins, size2_t size)
{
	std::vector<RES> res(begins.size() * size.x * size.y);

	RES* next = res.data();
	for (size2_t begin : begins)
	{
		copy_submatrix(src, next, src_size, begin, size);

		next += size.x * size.y;
	}

	return res;
}



}
