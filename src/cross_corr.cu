#include <cuda.h>
#include <cuda_runtime.h>

#include "kernels.cuh"
#include "device_launch_parameters.h"

namespace emida
{

/*  Optimization idea:
	The amount of work the threads have to do looks like this:
	64	72	80	88	96	88	80	72	64
	72	81	90	99	108	99	90	81	72
	80	90	100	110	120	110	100	90	80
	88	99	110	121	132	121	110	99	88
	96	108	120	132	144	132	120	108	96
	88	99	110	121	132	121	110	99	88
	80	90	100	110	120	110	100	90	80
	72	81	90	99	108	99	90	81	72
	64	72	80	88	96	88	80	72	64
	Threads near 0 offset do the most work.
	So some threads in the same warp/block may do much more work than others.
	Assign pixels to threads in a way that threads from the same thread do
	the same amount of work?
*/

template<typename T, typename RES>
__global__ void cross_corr(
	const T* __restrict__ pics,
	const T* __restrict__ ref,
	RES* __restrict__ res,
	size2_t size,
	size2_t res_size,
	size_t ref_slices,
	size_t batch_size)
{
	size_t whole_x = blockIdx.x * blockDim.x + threadIdx.x;
	size_t cuda_y = blockIdx.y * blockDim.y + threadIdx.y;
	
	//number of picture that this thread computes
	size_t slice_num = whole_x / res_size.x;

	if (slice_num >= ref_slices * batch_size || cuda_y >= res_size.y)
		return;

	size2_t slice_pos = { whole_x % res_size.x, cuda_y };
	size_t ref_num = slice_num % ref_slices;
	

	size2_t r = (res_size - 1) / 2;

	vec2<int> shift = { (int)slice_pos.x - (int)r.x, (int)slice_pos.y - (int)r.y };
	
	//change the pointers to point to the picture this thread computes
	pics += slice_num * size.area();
	ref += ref_num * size.area();
	res += slice_num * res_size.area();

	
	size_t x_end = shift.x < 0 ? size.x : size.x - shift.x;
	size_t y_end = shift.y < 0 ? size.y : size.y - shift.y;

	//control flow divergency in following fors??
	RES sum = 0;
	for (size_t y = shift.y >= 0 ? 0 : -shift.y; y < y_end; ++y)
	{
		for (size_t x = shift.x >= 0 ? 0 : -shift.x; x < x_end; ++x)
		{
			int x_shifted = x + shift.x;
			int y_shifted = y + shift.y;
			
			sum += pics[y_shifted * size.x + x_shifted] * ref[y * size.x + x];
		}
	}
	
	res[slice_pos.pos(res_size.x)] = sum;
}

template<typename T, typename RES>
void run_cross_corr(const T* pic_a, const T* pic_b, RES* res, vec2<size_t> size, vec2<size_t> res_size, size_t ref_slices, size_t batch_size)
{	
	dim3 block_size(16, 16);
	dim3 grid_size(div_up(res_size.x * batch_size * ref_slices, block_size.x), div_up(res_size.y, block_size.y));
	cross_corr<T, RES> <<<grid_size, block_size>>> (pic_a, pic_b, res, size, res_size, ref_slices, batch_size);
}


template void run_cross_corr<int, int>(
	const int*,
	const int*,
	int* res,
	vec2<size_t> size,
	vec2<size_t> res_size,
	size_t,
	size_t);

template void run_cross_corr<double, double>(
	const double*,
	const double*,
	double* res,
	vec2<size_t> size,
	vec2<size_t> res_size,
	size_t,
	size_t);

template void run_cross_corr<float, float>(
	const float*,
	const float*,
	float* res,
	vec2<size_t> size,
	vec2<size_t> res_size,
	size_t,
	size_t);




}