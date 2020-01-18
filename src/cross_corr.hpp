#pragma once

#include <cstdint>
#include <cmath>


namespace emida
{

template<typename T>
T point_diff(T& a, T& b)
{
	return a*b;
}

size_t idx(size_t cols, size_t x, size_t y)
{
	return y * cols + x;
}

template<typename T, typename RES>
void cross_corr_serial(const T* pic_a, const T* pic_b, RES * res, int cols, int rows)
{ 
	int res_cols = cols * 2 - 1;
	int res_rows = rows * 2 - 1;
	for (int x_shift = -cols + 1; x_shift < cols; ++x_shift)
	{
		for (int y_shift = -rows + 1; y_shift < rows; ++y_shift)
		{
			RES sum = 0;
			for (int y = 0; y < rows; ++y)
			{
				for (int x = 0; x < cols; ++x)
				{
					T b = 0;
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


}