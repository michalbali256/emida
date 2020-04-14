#pragma once

#include "common.hpp"

namespace emida {

template<typename T>
std::vector<T> sums_serial(const std::vector<T>& data, const std::vector<size2_t>& begins, size2_t src_size, size2_t slice_size)
{
	std::vector<T> res(begins.size());
	for (size_t i = 0; i < begins.size(); ++i)
	{
		for (size_t x = 0; x < slice_size.x; ++x)
			for (size_t y = 0; y < slice_size.y; ++y)
			{
				size2_t from = begins[i] + size2_t{ x, y };
				res[i] += data[from.pos(src_size.x)];
			}
	}
	return res;
}

}