#pragma once

#include <cstdint>
#include <cmath>
#include <vector>
#include <chrono>
#include <iostream>
#include <string>

#include "matrix.hpp"
#include "common.hpp"

namespace emida
{


inline size_t idx(size_t cols, size_t x, size_t y)
{
	return y * cols + x;
}

template<typename T>
std::vector<T> cross_corr_serial(const T* pic_a, const T* pic_b, size2_t pic_size)
{ 
	int res_cols = pic_size.x * 2 - 1;

	std::vector<T> res((pic_size * 2 - 1).area());
	int x_shift;
	for (x_shift = -(int)pic_size.x + 1; x_shift < (int)pic_size.x; ++x_shift)
	{
		for (int y_shift = -(int)pic_size.y + 1; y_shift < (int)pic_size.y; ++y_shift)
		{
			T sum = 0;
			for (int y = 0; y < pic_size.y; ++y)
			{
				for (int x = 0; x < pic_size.x; ++x)
				{
					int x_shifted = x + x_shift;
					int y_shifted = y + y_shift;
					if(x_shifted >= 0 && x_shifted < pic_size.x && y_shifted >= 0 && y_shifted < pic_size.y)
						sum += pic_a[idx(pic_size.x, x_shifted, y_shifted)] * pic_b[idx(pic_size.x, x, y)];
				}
			}

			res[idx(res_cols, x_shift + pic_size.x - 1, y_shift + pic_size.y - 1)] = sum;
		}
	}
	return res;
}




}