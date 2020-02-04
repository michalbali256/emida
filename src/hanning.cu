#include "cuda.h"
#include "cuda_runtime.h"

#include "kernels.cuh"

namespace emida
{

template<typename T>
__global__ void hanning(T* __restrict pic, const T* hanning_x, const T* hanning_y, int cols, int rows)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= cols || y >= rows)
		return;

	pic[y * cols + x] = pic[y * cols + x] * hanning_x[x] * hanning_y[y];
}

template<typename T>
void run_hanning(T* pic, const T* hanning_x, const T* hanning_y, int cols, int rows)
{	
	dim3 block_size(16, 16);
	dim3 grid_size(div_up(cols, block_size.x), div_up(rows, block_size.y));
	hanning<T> <<<grid_size, block_size>>> (pic, hanning_x, hanning_y, cols, rows);
}


template void run_hanning<double>(double * pic, const double * hanning_x, const double* hanning_y, int cols, int rows);

}