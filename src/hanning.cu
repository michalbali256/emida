#include "cuda.h"
#include "cuda_runtime.h"

namespace emida
{

template<typename T>
__global__ void hanning(T* __restrict pic, const T* hanning_x, const T* hanning_y, int cols, int rows)
{
	int res_cols = cols * 2 - 1;
	
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (cuda_x >= cols || cuda_y > y_shift)
		return;

	pic[y * cols + x] = pic[y * cols + x] * hanning_x[x] * hanning_y[y];
}

template<typename T>
void run_hanning(T* pic, const T* hanning_x, const T* hanning_y, int cols, int rows)
{	
	dim3 block_size(16, 16);
	dim3 grid_size(div_up(2 * cols, block_size.x), div_up(2 * rows, block_size.y));
	hanning<T, RES> <<<grid_size, block_size>>> (pic_a, pic_b, res, cols, rows);
}


template void run_hanning<double>(double * pic, double T* hanning_x, const double* hanning_y, int cols, int rows);

}