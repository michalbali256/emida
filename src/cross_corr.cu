#include <cuda.h>
#include <cuda_runtime.h>

#include "kernels.cuh"
#include "device_launch_parameters.h"
#include "device_helpers.hpp"

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

	if (slice_num >= ref_slices || cuda_y >= res_size.y)
		return;

	size2_t slice_pos = { whole_x % res_size.x, cuda_y };
	size_t ref_num = slice_num % ref_slices;
	

	size2_t r = (res_size - 1) / 2;

	vec2<int> shift = { (int)slice_pos.x - (int)r.x, (int)slice_pos.y - (int)r.y };
	
	ref += ref_num * size.area();
	pics += slice_num * size.area();
	res += slice_num * res_size.area();

	
	for (size_t i = 0; i < batch_size; ++i)
	{
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

		pics += ref_slices * size.area();
		res += ref_slices * res_size.area();
	}
}

template<typename T, typename RES>
void run_cross_corr(const T* pic_a, const T* pic_b, RES* res, vec2<size_t> size, vec2<size_t> res_size, size_t ref_slices, size_t batch_size)
{	
	dim3 block_size(16, 16);
	dim3 grid_size(div_up(res_size.x * ref_slices, block_size.x), div_up(res_size.y, block_size.y));
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

template<typename T>
__device__ __inline__ void copy_subregion(const T * __restrict__ src, size_t src_stride, T* __restrict__ dest, size2_t dest_size, size2_t region_pos)
{
	for (size_t x = threadIdx.x; x < dest_size.x; x += blockDim.x)
		for (size_t y = threadIdx.y; y < dest_size.y; y += blockDim.y)
		{
			dest[y * dest_size.x + x] = src[(y + region_pos.y) * src_stride + (x + region_pos.x)];
		}
}

template<typename T, typename RES>
__global__ void cross_corr_opt(
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

	size_t block_grid_width = div_up(size.x, gridDim.x);
	size2_t pic_block_pos = size2_t::from_id(blockIdx.x, block_grid_width) * blockDim;
	size2_t ref_block_pos = size2_t::from_id(blockIdx.y, block_grid_width) * blockDim;

	size2_t reg_size = { blockDim.x / 2, blockDim.y / 2 };
	size2_t res_reg_size = { blockDim.x - 1, blockDim.y - 1 };


	T* smem = shared_memory_proxy<T>();
	T* pic_reg = smem;
	T* ref_reg = smem + reg_size.area();
	T* res_reg = smem + 2 * + reg_size.area();

	copy_subregion(pics, size.x, pic_reg, size2_t{ blockDim.x, blockDim.y }, pic_block_pos);
	copy_subregion(ref, size.x, pic_reg, size2_t{ blockDim.x, blockDim.y }, pic_block_pos);

	__syncthreads();

	size2_t r = (res_size - 1) / 2;
	size2_t reg_r = reg_size - 1;

	vec2<int> shift = { (int)threadIdx.x - (int)reg_r.x, (int)threadIdx.y - (int)reg_r.y };

	//ref += ref_num * size.area();
	//pics += slice_num * size.area();
	//res += slice_num * res_size.area();


	size_t x_end = shift.x < 0 ? blockDim.x : blockDim.x - shift.x;
	size_t y_end = shift.y < 0 ? blockDim.y : blockDim.y - shift.y;

	//control flow divergency in following fors??
	RES sum = 0;
	for (size_t y = shift.y >= 0 ? 0 : -shift.y; y < y_end; ++y)
	{
		for (size_t x = shift.x >= 0 ? 0 : -shift.x; x < x_end; ++x)
		{
			int x_shifted = x + shift.x;
			int y_shifted = y + shift.y;

			sum += pic_reg[y_shifted * reg_size.x + x_shifted] * ref_reg[y * reg_size.x + x];
		}
	}

	
	RES* res_ptr = res + (pic_block_pos - ref_block_pos + size2_t{ threadIdx.x, threadIdx.y }).pos(res_size.x);
	atomicAdd(res_ptr, sum);

	//pics += ref_slices * size.area();
	//res += ref_slices * res_size.area();

}

template<typename T, typename RES>
void run_cross_corr_opt(
	const T* pic_a,
	const T* pic_b,
	RES* res,
	size2_t size,
	size2_t res_size,
	size2_t block_size,
	size_t ref_slices,
	size_t batch_size)
{
	//smem: 6 * blockDim.x * blockDim.y
	dim3 block_dim(block_size.x, block_size.y);
	size_t blocks = div_up(size.x, block_size.x) * div_up(size.y, block_size.y);
	dim3 grid_size(blocks, blocks);
	cross_corr_opt<T, RES> <<<grid_size, block_dim, 2 * block_size.area() >>> (pic_a, pic_b, res, size, res_size, ref_slices, batch_size);
}

template void run_cross_corr_opt<double, double>(
	const double*,
	const double*,
	double* res,
	size2_t size,
	size2_t res_size,
	size2_t block_size,
	size_t,
	size_t);


}