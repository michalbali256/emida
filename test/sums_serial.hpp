#pragma once

#include "common.hpp"

namespace emida {

template<typename T>
std::vector<double> sums_serial(const std::vector<T>& data,
	const std::vector<size2_t>& begins,
	size2_t src_size,
	size2_t slice_size,
	size_t batch_size = 1)
{
	std::vector<double> res(begins.size() * batch_size);

	for (size_t b = 0; b < batch_size; ++b)
	{
		double* res_b = res.data() + b * begins.size();
		const T* data_b = data.data() + b * src_size.area();

		for (size_t i = 0; i < begins.size(); ++i)
		{
			for (size_t x = 0; x < slice_size.x; ++x)
				for (size_t y = 0; y < slice_size.y; ++y)
				{
					size2_t from = begins[i] + size2_t{ x, y };
					res_b[i] += data_b[from.pos(src_size.x)];
				}
		}
	}
	return res;
}

}