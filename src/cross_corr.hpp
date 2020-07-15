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
std::vector<T> cross_corr_serial(const T* pics, const T* refs, size2_t pic_size, size_t ref_slices)
{
	size2_t res_size = (pic_size * 2 - 1);
	return cross_corr_serial(pics, refs, pic_size, res_size, ref_slices);
}

template<typename T>
std::vector<T> cross_corr_serial(const T* pics, const T* refs, size2_t pic_size, size2_t res_size, size_t ref_slices)
{ 
	
	size2_t res_r = (res_size - 1) / 2;

	std::vector<T> res_vector(res_size.area() * ref_slices);
	T* res = res_vector.data();

	for (size_t i = 0; i < ref_slices; ++i)
	{
		for (int x_shift = -(int)res_r.x; x_shift <= (int)res_r.x; ++x_shift)
		{
			for (int y_shift = -(int)res_r.y; y_shift <= (int)res_r.y; ++y_shift)
			{
				T sum = 0;
				for (int y = 0; y < (int)pic_size.y; ++y)
				{
					for (int x = 0; x < (int)pic_size.x; ++x)
					{
						int x_shifted = x + x_shift;
						int y_shifted = y + y_shift;
						if (x_shifted >= 0 && x_shifted < (int)pic_size.x && y_shifted >= 0 && y_shifted < (int)pic_size.y)
							sum += pics[idx(pic_size.x, x_shifted, y_shifted)] * refs[idx(pic_size.x, x, y)];
					}
				}

				res[idx(res_size.x, x_shift + res_r.x, y_shift + res_r.y)] = sum;
			}
		}

		res += res_size.area();
		pics += pic_size.area();
		refs += pic_size.area();
	}
	return res_vector;
}




}