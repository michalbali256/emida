#include <cuda.h>
#include <cuda_runtime.h>

#include "kernels.cuh"
#include "device_launch_parameters.h"

namespace emida
{

template<typename T, typename RES>
__global__ void cross_corr(const T* pics_a, const T* pics_b, RES* res, size_t cols, size_t rows, size_t batch_size)
{
	size_t res_cols = cols * 2 - 1;
	size_t res_rows = rows * 2 - 1;
	size_t res_size = res_cols * res_rows;

	size_t pic_size = cols * rows;

	size_t whole_x = blockIdx.x * blockDim.x + threadIdx.x;
	size_t cuda_y = blockIdx.y * blockDim.y + threadIdx.y;
	
	size_t res_x = whole_x % res_cols;
	size_t pic_num = whole_x / res_cols;
	

	int x_shift = res_x - cols + 1;
	int y_shift = cuda_y - rows + 1;

	if (pic_num >= batch_size || cuda_y >= res_rows)
		return;
	
	//change the pointers to point to the picture this thread computes
	pics_a += pic_num * pic_size;
	pics_b += pic_num * pic_size;
	res += pic_num * res_size;

	RES sum = 0;
	for (size_t y = 0; y < rows; ++y)
	{
		for (size_t x = 0; x < cols; ++x)
		{
			int x_shifted = x + x_shift;
			int y_shifted = y + y_shift;
			if (x_shifted >= 0 && x_shifted < cols && y_shifted >= 0 && y_shifted < rows)
				sum += pics_a[y_shifted * cols + x_shifted] * pics_b[y * cols + x];
		}
	}

	res[(y_shift + rows - 1) * res_cols + x_shift + cols - 1] = sum;
}

template<typename T, typename RES>
void run_cross_corr(const T* pic_a, const T* pic_b, RES* res, size_t cols, size_t rows, size_t batch_size)
{	
	dim3 block_size(16, 16);
	dim3 grid_size(div_up((2 * cols - 1) * batch_size, block_size.x), div_up(2 * rows - 1, block_size.y));
	cross_corr<T, RES> <<<grid_size, block_size>>> (pic_a, pic_b, res, cols, rows, batch_size);
}


template void run_cross_corr<int, int>(const int*, const int*, int* res, size_t, size_t, size_t);
template void run_cross_corr<double, double>(const double*, const double*, double* res, size_t, size_t, size_t);

}