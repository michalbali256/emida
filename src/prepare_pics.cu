#include "cuda.h"
#include "cuda_runtime.h"

#include "kernels.cuh"

namespace emida
{

template<typename T>
__global__ void prepare_pics(T* __restrict pic, const T* hanning_x, const T* hanning_y, const T* sums, size2_t size, size_t batch_size)
{
	size_t whole_x = blockIdx.x * blockDim.x + threadIdx.x;
	size_t pic_x = whole_x % size.x;
	size_t pic_num = whole_x / size.x;
	size_t y = blockIdx.y * blockDim.y + threadIdx.y;

	//Problem? many threads may end up nothing if number of rows and block_size.y are in poor combination
	if (pic_num >= batch_size || y >= size.y)
		return;

	T& pixel = pic[pic_num * size.area() + y * size.x + pic_x];
	//subtract mean of the picture
	pixel -= sums[pic_num] / size.area();
	//apply hanning filter
	pixel *= hanning_x[pic_x] * hanning_y[y];
}

template<typename T>
void run_prepare_pics(T* pic, const T* hanning_x, const T* hanning_y, const T * sums, size2_t size, size_t batch_size)
{	
	dim3 block_size(16, 16);
	dim3 grid_size(div_up(size.x * batch_size, block_size.x), div_up(size.y, block_size.y));
	prepare_pics<T> <<<grid_size, block_size>>> (pic, hanning_x, hanning_y, sums, size, batch_size);
}


template void run_prepare_pics<double>(double * pic, const double * hanning_x,
	const double* hanning_y, const double * sums, size2_t size, size_t batch_size);

}