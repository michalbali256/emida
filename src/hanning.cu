#include "cuda.h"
#include "cuda_runtime.h"

#include "kernels.cuh"

namespace emida
{

template<typename T>
__global__ void hanning(T* __restrict pic, const T* hanning_x, const T* hanning_y, size_t cols, size_t rows, size_t batch_size)
{
	size_t pic_size = cols * rows;
	size_t whole_x = blockIdx.x * blockDim.x + threadIdx.x;
	size_t pic_x = whole_x % cols;
	size_t pic_num = whole_x / cols;
	size_t y = blockIdx.y * blockDim.y + threadIdx.y;

	//Problem? many threads may end up nothing if number of rows and block_size.y are in poor combination
	if (pic_num >= batch_size || y >= rows)
		return;

	pic[pic_num * pic_size + y * cols + pic_x] *= hanning_x[pic_x] * hanning_y[y];
}

template<typename T>
void run_hanning(T* pic, const T* hanning_x, const T* hanning_y, size_t cols, size_t rows, size_t batch_size)
{	
	dim3 block_size(16, 16);
	dim3 grid_size(div_up(cols * batch_size, block_size.x), div_up(rows, block_size.y));
	hanning<T> <<<grid_size, block_size>>> (pic, hanning_x, hanning_y, cols, rows, batch_size);
}


template void run_hanning<double>(double * pic, const double * hanning_x,
	const double* hanning_y, size_t cols, size_t rows, size_t batch_size);

}