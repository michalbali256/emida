#pragma once

#include <cstdint>
#include <cmath>
#include <vector>
#include <chrono>
#include <iostream>
#include <string>

#include "matrix.hpp"

namespace emida
{

template<typename T>
inline T point_diff(T& a, T& b)
{
	return a*b;
}

inline size_t idx(size_t cols, size_t x, size_t y)
{
	return y * cols + x;
}

template<typename T, typename RES>
void cross_corr_serial(const T* pic_a, const T* pic_b, RES * res, int cols, int rows)
{ 
	int res_cols = cols * 2 - 1;

	for (int x_shift = -cols + 1; x_shift < cols; ++x_shift)
	{
		for (int y_shift = -rows + 1; y_shift < rows; ++y_shift)
		{
			RES sum = 0;
			for (int y = 0; y < rows; ++y)
			{
				for (int x = 0; x < cols; ++x)
				{
					int x_shifted = x + x_shift;
					int y_shifted = y + y_shift;
					if(x_shifted >= 0 && x_shifted < cols && y_shifted >= 0 && y_shifted < rows)
						sum += point_diff(pic_a[idx(cols, x, y)], pic_b[idx(cols, x_shifted, y_shifted)]);
				}
			}

			res[idx(res_cols, x_shift + cols - 1, y_shift + rows - 1)] = sum;
		}
	}

}


inline std::vector<int> do_serial(const matrix<int> & a, const matrix<int> & b)
{
	size_t res_n = 2 * a.n - 1;
	std::vector<int> res;
	res.resize(res_n * res_n);
	std::chrono::high_resolution_clock c;
	auto start = c.now();
	emida::cross_corr_serial<int, int>(b.data.data(), a.data.data(), res.data(), a.n, a.n);
	auto end = c.now();
	std::chrono::duration<double> dur = end - start;
	auto dur_milli = std::chrono::duration_cast<std::chrono::milliseconds>(dur);
	std::cout << "Cross corr duration: " << std::to_string(dur_milli.count()) << " ms" << "\n";

	/*for (size_t i = 0; i < res_n; ++i)
{
	for (size_t j = 0; j < res_n; ++j)
	{
		std::cout << res[i * res_n + j] << "\t";
	}
	std::cout << "\n";
}*/

	return res;
}

}