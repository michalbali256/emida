#include <cuda.h>
#include <cuda_runtime.h>

#include "kernels.cuh"
#include "device_launch_parameters.h"

namespace emida
{

template<typename T, typename RES>
__global__ void cross_corr(const T* pics_a, const T* pics_b, RES* res, vec2<size_t> size, vec2<size_t> res_size, size_t batch_size)
{
	size_t pic_size = size.area();

	size_t whole_x = blockIdx.x * blockDim.x + threadIdx.x;
	size_t cuda_y = blockIdx.y * blockDim.y + threadIdx.y;
	
	//number of picture that this thread computes
	size_t pic_num = whole_x / res_size.x;

	//x in that picture
	size_t res_x = whole_x % res_size.x;

	vec2<size_t> r = (res_size - 1) / 2;

	int x_shift = res_x - r.x;
	int y_shift = cuda_y - r.y;

	if (pic_num >= batch_size || cuda_y >= res_size.y)
		return;
	
	//change the pointers to point to the picture this thread computes
	pics_a += pic_num * pic_size;
	pics_b += pic_num * pic_size;
	res += pic_num * res_size.area();

	

	RES sum = 0;
	for (size_t y = 0; y < size.y; ++y)
	{
		for (size_t x = 0; x < size.x; ++x)
		{
			int x_shifted = x + x_shift;
			int y_shifted = y + y_shift;
			if (x_shifted >= 0 && x_shifted < size.x && y_shifted >= 0 && y_shifted < size.y)
				sum += pics_a[y_shifted * size.x + x_shifted] * pics_b[y * size.x + x];
		}
	}

	res[cuda_y * res_size.x + res_x] = sum;
}

template<typename T, typename RES>
void run_cross_corr(const T* pic_a, const T* pic_b, RES* res, vec2<size_t> size, vec2<size_t> res_size, size_t batch_size)
{	
	dim3 block_size(16, 16);
	dim3 grid_size(div_up(res_size.x * batch_size, block_size.x), div_up(res_size.y * batch_size, block_size.y));
	cross_corr<T, RES> <<<grid_size, block_size>>> (pic_a, pic_b, res, size, res_size, batch_size);
}


template void run_cross_corr<int, int>(const int*, const int*, int* res, vec2<size_t> size, vec2<size_t> res_size, size_t);
template void run_cross_corr<double, double>(const double*, const double*, double* res, vec2<size_t> size, vec2<size_t> res_size, size_t);

}