#include "cuda.h"
#include "cuda_runtime.h"

namespace emida
{

template<typename T, typename RES>
__global__ void cross_corr(const T* pic_a, const T* pic_b, RES* res, int cols, int rows)
{
	int res_cols = cols * 2 - 1;
	
	int cuda_x = blockIdx.x * blockDim.x + threadIdx.x;
	int cuda_y = blockIdx.y * blockDim.y + threadIdx.y;

	int x_shift = cuda_x - cols + 1;

	int y_shift = cuda_y - rows + 1;
	if (x_shift >= cols || y_shift > y_shift)
		return;

	RES sum = 0;
	for (int y = 0; y < rows; ++y)
	{
		for (int x = 0; x < cols; ++x)
		{
			int x_shifted = x + x_shift;
			int y_shifted = y + y_shift;
			if (x_shifted >= 0 && x_shifted < cols && y_shifted >= 0 && y_shifted < rows)
				sum += pic_a[y*cols + x] * pic_b[y_shifted * cols + x_shifted];
		}
	}

	res[(y_shift + rows - 1) * res_cols + x_shift + cols - 1] = sum;
}

template<typename T, typename U>
T div_up(T a, U b)
{
	return (a + b - 1) / b;
}

template<typename T, typename RES>
void run_cross_corr(const T* pic_a, const T* pic_b, RES* res, int cols, int rows)
{	
	dim3 block_size(16, 16);
	dim3 grid_size(div_up(2 * cols, block_size.x), div_up(2 * rows, block_size.y));
	cross_corr<T, RES> <<<grid_size, block_size>>> (pic_a, pic_b, res, cols, rows);
}


template void run_cross_corr<int, int>(const int*, const int*, int* res, int, int);

}