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
	esize_t ref_slices,
	esize_t batch_size)
{
	esize_t whole_x = blockIdx.x * blockDim.x + threadIdx.x;
	esize_t cuda_y = blockIdx.y * blockDim.y + threadIdx.y;
	
	//number of picture that this thread computes
	esize_t slice_num = whole_x / res_size.x;

	if (slice_num >= ref_slices || cuda_y >= res_size.y)
		return;

	size2_t slice_pos = { whole_x % res_size.x, cuda_y };
	esize_t ref_num = slice_num % ref_slices;
	

	size2_t r = (res_size - 1) / 2;

	vec2<int> shift = { (int)slice_pos.x - (int)r.x, (int)slice_pos.y - (int)r.y };
	
	ref += ref_num * size.area();
	pics += slice_num * size.area();
	res += slice_num * res_size.area();

	
	for (esize_t i = 0; i < batch_size; ++i)
	{
		esize_t x_end = min(size.x - shift.x, size.x);// shift.x < 0 ? size.x : size.x - shift.x;
		esize_t y_end = min(size.y - shift.y, size.y);//shift.y < 0 ? size.y : size.y - shift.y;

		//control flow divergency in following fors??
		RES sum = 0;
		for (esize_t y = max(-shift.y, 0); y < y_end; ++y)
		{
			for (esize_t x = max(-shift.x, 0); x < x_end; ++x)
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
void run_cross_corr(const T* pic_a, const T* pic_b, RES* res, size2_t size, size2_t res_size, esize_t ref_slices, esize_t batch_size)
{	
	dim3 block_size(16, 16);
	dim3 grid_size(div_up(res_size.x * ref_slices, block_size.x), div_up(res_size.y, block_size.y));
	cross_corr<T, RES> <<<grid_size, block_size>>> (pic_a, pic_b, res, size, res_size, ref_slices, batch_size);
}


template void run_cross_corr<int, int>(
	const int*,
	const int*,
	int* res,
	size2_t size,
	size2_t res_size,
	esize_t,
	esize_t);

template void run_cross_corr<double, double>(
	const double*,
	const double*,
	double* res,
	size2_t size,
	size2_t res_size,
	esize_t,
	esize_t);

template void run_cross_corr<float, float>(
	const float*,
	const float*,
	float* res,
	size2_t size,
	size2_t res_size,
	esize_t,
	esize_t);

//*******************************************************************************************************************************************************

template<int k, typename T, typename RES>
__device__ __inline__ void compute(const T* __restrict__ pics, const T* __restrict__ ref, RES* __restrict__ res, size2_t size, size2_t res_size, size2_t slice_pos, vec2<int> shift)
{
	esize_t x_end = shift.x < 0 ? size.x : size.x - shift.x;
	esize_t y_end = shift.y < 0 ? size.y : size.y - shift.y;

	//control flow divergency in following fors??
	RES sum[k];

	#pragma unroll
	for (esize_t i = 0; i < k; ++i)
		sum[i] = 0;
	for (esize_t y = shift.y >= 0 ? 0 : -shift.y; y < y_end; ++y)
	{
		for (esize_t x = shift.x >= 0 ? 0 : -shift.x; x < x_end; ++x)
		{
			int x_shifted = x + shift.x;
			int y_shifted = y + shift.y;
			
			
			for (esize_t i = 0; i < k; ++i)
				sum[i] += pics[y_shifted * size.x + x_shifted + i] * ref[y * size.x + x];
		}
	}

	#pragma unroll
	for (esize_t i = 0; i < k; ++i)
		res[slice_pos.y * res_size.x + slice_pos.x + i] = sum[i];
}

template<>
__device__ __inline__ void compute<2, double, double>(const double* __restrict__ pics, const double* __restrict__ ref, double* __restrict__ res, size2_t size, size2_t res_size, size2_t slice_pos, vec2<int> shift)
{
	esize_t x_start = shift.x < 0 ? -shift.x : 0;
	esize_t x_end = shift.x < 0 ? size.x : size.x - shift.x;
	esize_t y_end = shift.y < 0 ? size.y : size.y - shift.y;

	//control flow divergency in following fors??
	double sum[2];

	sum[0] = 0;
	sum[1] = 0;
	

	

	for (esize_t y = shift.y >= 0 ? 0 : -shift.y; y < y_end; ++y)
	{
		int y_shifted = y + shift.y;
		double cach[2];
		cach[0] = pics[y_shifted * size.x + x_start + shift.x];
		int x_shifted = x_start + shift.x;
		for (esize_t x = x_start; x < x_end; x+=2)
		{
			++x_shifted;
			cach[1] = pics[y_shifted * size.x + x_shifted];
			sum[0] += cach[0] * ref[y * size.x + x];
			sum[1] += cach[1] * ref[y * size.x + x];
			
			++x_shifted;
			cach[0] = pics[y_shifted * size.x + x_shifted];
			sum[0] += cach[1] * ref[y * size.x + x + 1];
			sum[1] += cach[0] * ref[y * size.x + x + 1];
			
		}
	}

	res[slice_pos.y * res_size.x + slice_pos.x] = sum[0];
	res[slice_pos.y * res_size.x + slice_pos.x + 1] = sum[1];
}

template<>
__device__ __inline__ void compute<3, double, double>(const double* __restrict__ pics, const double* __restrict__ ref, double* __restrict__ res, size2_t size, size2_t res_size, size2_t slice_pos, vec2<int> shift)
{
	esize_t x_end = shift.x < 0 ? size.x : size.x - shift.x;
	esize_t y_end = shift.y < 0 ? size.y : size.y - shift.y;

	//control flow divergency in following fors??
	double sum[3];

	sum[0] = 0;
	sum[1] = 0;
	sum[3] = 0;


	esize_t x_start = shift.x >= 0 ? 0 : -shift.x;

	for (esize_t y = shift.y >= 0 ? 0 : -shift.y; y < y_end; ++y)
	{
		int y_shifted = y + shift.y;
		double cach[3];
		cach[0] = pics[y_shifted * size.x + x_start + shift.x];
		cach[1] = pics[y_shifted * size.x + x_start + shift.x + 1];
		int x_shifted = x_start + shift.x;
		for (esize_t x = x_start; x < x_end; x += 3)
		{
			++x_shifted;
			cach[2] = pics[y_shifted * size.x + x_shifted];
			//sum[0] += pics[y_shifted * size.x + x_shifted] * ref[y * size.x + x];
			//sum[1] += pics[y_shifted * size.x + x_shifted + 1] * ref[y * size.x + x];
			sum[0] += cach[0] * ref[y * size.x + x];
			sum[1] += cach[1] * ref[y * size.x + x];
			sum[2] += cach[2] * ref[y * size.x + x];

			++x_shifted;
			cach[0] = pics[y_shifted * size.x + x_shifted];
			sum[0] += cach[1] * ref[y * size.x + x + 1];
			sum[1] += cach[2] * ref[y * size.x + x + 1];
			sum[2] += cach[0] * ref[y * size.x + x + 1];

			++x_shifted;
			cach[1] = pics[y_shifted * size.x + x_shifted];
			sum[0] += cach[2] * ref[y * size.x + x + 2];
			sum[1] += cach[0] * ref[y * size.x + x + 2];
			sum[2] += cach[1] * ref[y * size.x + x + 2];

		}
	}

	res[slice_pos.y * res_size.x + slice_pos.x] = sum[0];
	res[slice_pos.y * res_size.x + slice_pos.x + 1] = sum[1];
	res[slice_pos.y * res_size.x + slice_pos.x + 2] = sum[2];
}


template<typename T, typename RES>
__device__ __inline__ void compute_dyn(const T* __restrict__ pics, const T* __restrict__ ref, RES* __restrict__ res, size2_t size, size2_t res_size, size2_t slice_pos, vec2<int> shift, int k)
{
	switch (k)
	{
	case 1:
		compute<1>(pics, ref, res, size, res_size, slice_pos, shift);
		break;
	case 2:
		compute<2>(pics, ref, res, size, res_size, slice_pos, shift);
		break;
	case 3:
		compute<3>(pics, ref, res, size, res_size, slice_pos, shift);
		break;
	case 4:
		compute<4>(pics, ref, res, size, res_size, slice_pos, shift);
		break;
	case 5:
		compute<5>(pics, ref, res, size, res_size, slice_pos, shift);
		break;
	case 6:
		compute<6>(pics, ref, res, size, res_size, slice_pos, shift);
		break;
	default:
		printf("%d", k);
	}
}


template<int k, typename T, typename RES>
__global__ void cross_corr_r(
	const T* __restrict__ pics,
	const T* __restrict__ ref,
	RES* __restrict__ res,
	size2_t size,
	size2_t res_size,
	esize_t ref_slices,
	esize_t batch_size)
{
	esize_t whole_x = blockIdx.x * blockDim.x + threadIdx.x;
	esize_t cuda_y = blockIdx.y * blockDim.y + threadIdx.y;

	//number of picture that this thread computes
	esize_t slice_num = whole_x / res_size.x;

	if (slice_num >= ref_slices || cuda_y >= res_size.y)
		return;

	size2_t slice_pos = { (whole_x % div_up(res_size.x, k))*k, cuda_y };
	esize_t ref_num = slice_num % ref_slices;

	

	size2_t r = (res_size - 1) / 2;

	vec2<int> shift = { (int)slice_pos.x - (int)r.x, (int)slice_pos.y - (int)r.y };

	ref += ref_num * size.area();
	pics += slice_num * size.area();
	res += slice_num * res_size.area();

	

	for (esize_t i = 0; i < batch_size; ++i)
	{
		//printf("[%d %d] %d %d\n", (int)slice_pos.x, (int)slice_pos.y, k, (int)res_size.x);
		if ((int)slice_pos.x + k > (int)res_size.x)
			compute_dyn(pics, ref, res, size, res_size, slice_pos, shift, (int)res_size.x - (int)slice_pos.x);
		else
			compute<k, T, RES>(pics, ref, res, size, res_size, slice_pos, shift);

		pics += ref_slices * size.area();
		res += ref_slices * res_size.area();
	}
}

template<typename T, typename RES>
void run_cross_corr_r(const T* pic_a, const T* pic_b, RES* res, size2_t size, size2_t res_size, esize_t ref_slices, esize_t batch_size)
{
	constexpr int k = 2;
	dim3 block_size(16, 16);
	dim3 grid_size(div_up(div_up(res_size.x, k) * ref_slices, block_size.x), div_up(res_size.y, block_size.y));
	cross_corr_r<k, T, RES> <<<grid_size, block_size >>> (pic_a, pic_b, res, size, res_size, ref_slices, batch_size);
}

template void run_cross_corr_r<double, double>(
	const double*,
	const double*,
	double* res,
	size2_t size,
	size2_t res_size,
	esize_t,
	esize_t);

template void run_cross_corr_r<float, float>(
	const float*,
	const float*,
	float* res,
	size2_t size,
	size2_t res_size,
	esize_t,
	esize_t);


//*******************************************************************************************************************************************************
template<typename T>
__device__ __inline__ void copy_subregion(const T * __restrict__ src, size2_t src_size, T* __restrict__ dest, size2_t dest_size, size2_t region_pos)
{
	for (esize_t y = threadIdx.y; y < dest_size.y; y += blockDim.y)
		for (esize_t x = threadIdx.x; x < dest_size.x; x += blockDim.x)
		{
			dest[y * dest_size.x + x] = x + region_pos.x < src_size.x && y + region_pos.y < src_size.y
				? src[(y + region_pos.y) * src_size.x + (x + region_pos.x)]
				: 0;
		}
}

template<typename T, typename RES>
__global__ void cross_corr_nopt(
	const T* __restrict__ pics,
	const T* __restrict__ ref,
	RES* __restrict__ res,
	size2_t size,
	size2_t res_size,
	esize_t ref_slices,
	esize_t batch_size)
{
	size2_t reg_size = { (blockDim.x + 1) / 2, (blockDim.y + 1) / 2 };

	esize_t block_grid_width = div_up(size.x, reg_size.x);
	
	esize_t one_slice_blocks = gridDim.x / ref_slices;
	esize_t slice_num = blockIdx.x / one_slice_blocks;
	esize_t block_idx_x = blockIdx.x % one_slice_blocks;

	size2_t pic_block_pos = size2_t::from_id(block_idx_x, block_grid_width) * reg_size;
	size2_t ref_block_pos = size2_t::from_id(blockIdx.y, block_grid_width) * reg_size;
	/*if (threadIdx.x == 0 && threadIdx.y == 0)
	{
		//printf("%d %d %d %d\n", blockDim.x, blockDim.y, gridDim.x, gridDim.y);

		printf("%d %d %d %d %d %d\n", blockIdx.x, (int)block_grid_width, (int)pic_block_pos.x, (int)pic_block_pos.y, (int)ref_block_pos.x, (int)ref_block_pos.y);
	}__syncthreads();*/
	
	T* smem = shared_memory_proxy<T>();
	T* pic_reg = smem;
	T* ref_reg = smem + reg_size.area();
	T* res_reg = smem + 2 * + reg_size.area();

	//ref += ref_num * size.area();
	ref += slice_num * size.area();
	pics += slice_num * size.area();
	res += slice_num * res_size.area();

	copy_subregion(pics, size, pic_reg, reg_size, pic_block_pos);
	copy_subregion(ref, size, ref_reg, reg_size, ref_block_pos);
	__syncthreads();
	/*if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 2 && blockIdx.y == 0)
	{
		printf("bb %d %d\n", (int)pic_block_pos.x, (int)pic_block_pos.y);
		for (esize_t i = 0; i < reg_size.y; ++i)
		{
			for (esize_t j = 0; j < reg_size.x; j++)
			{
				printf("%f ", pic_reg[i * reg_size.x + j]);
			}
			printf("\n");
		}
		for (esize_t i = 0; i < reg_size.y; ++i)
		{
			for (esize_t j = 0; j < reg_size.x; j++)
			{
				printf("%f ", ref_reg[i * reg_size.x + j]);
			}
			printf("\n");
		}
		
	}__syncthreads();*/

	//size2_t r = (res_size - 1) / 2;
	size2_t reg_r = reg_size - 1;
	vec2<int> res_r = { ((int)res_size.x - 1) / 2, ((int)res_size.y - 1) / 2 };
	vec2<int> shift = { (int)threadIdx.x - (int)reg_r.x, (int)threadIdx.y - (int)reg_r.y };
	vec2<int> block_shift = { (int)pic_block_pos.x - (int)ref_block_pos.x, (int)pic_block_pos.y - (int)ref_block_pos.y };
	vec2<int> res_pos = block_shift + shift + res_r;
	if (res_pos.x >= res_size.x || res_pos.y >= res_size.y)
		return;
	RES* res_ptr = res + (res_pos).pos(res_size.x);


	esize_t x_end = shift.x < 0 ? reg_size.x : reg_size.x - shift.x;
	esize_t y_end = shift.y < 0 ? reg_size.y : reg_size.y - shift.y;

	//control flow divergency in following fors??
	RES sum = 0;
	for (esize_t y = shift.y >= 0 ? 0 : -shift.y; y < y_end; ++y)
	{
		for (esize_t x = shift.x >= 0 ? 0 : -shift.x; x < x_end; ++x)
		{
			int x_shifted = x + shift.x;
			int y_shifted = y + shift.y;

			sum += pic_reg[y_shifted * reg_size.x + x_shifted] * ref_reg[y * reg_size.x + x];
		}
	}
	

	//vec2<int> block_shift = { (int)ref_block_pos.x - (int)pic_block_pos.x, (int)ref_block_pos.y - (int)pic_block_pos.y };

	//printf("%d %d %d %d %d %d %f %d %d %d %d %d %d\n", threadIdx.x, threadIdx.y, (int)pic_block_pos.x, (int)pic_block_pos.y, (int)ref_block_pos.x, (int)ref_block_pos.y, sum, (block_shift + shift + res_r).x,(block_shift + shift + res_r).y, shift.x, shift.y, res_r.x, res_r.y);
	
	atomicAdd(res_ptr, sum);
	

	//pics += ref_slices * size.area();
	//res += ref_slices * res_size.area();

}

template<typename T, typename RES>
void run_cross_corr_nopt(
	const T* pics,
	const T* ref,
	RES* res,
	size2_t size,
	size2_t res_size,
	size2_t block_size,
	esize_t ref_slices,
	esize_t batch_size)
{
	dim3 block_dim(block_size.x, block_size.y);
	size2_t in_block_size = (block_size + 1) / 2;
	esize_t blocks = div_up(size.x, in_block_size.x) * div_up(size.y, in_block_size.y);
	dim3 grid_size(blocks * ref_slices, blocks);
	cross_corr_nopt<T, RES> <<<grid_size, block_dim, 2 * in_block_size.area() * sizeof(T) >>> (pics, ref, res, size, res_size, ref_slices, batch_size);
}

template void run_cross_corr_nopt<double, double>(
	const double*,
	const double*,
	double* res,
	size2_t size,
	size2_t res_size,
	size2_t block_size,
	esize_t,
	esize_t);

template void run_cross_corr_nopt<float, float>(
	const float*,
	const float*,
	float* res,
	size2_t size,
	size2_t res_size,
	size2_t block_size,
	esize_t,
	esize_t);


//*******************************************************************************************************************************************************
constexpr int stripe_size = 8;

template<typename T, typename RES>
__global__ void cross_corr_opt(
	const T* __restrict__ pics,
	const T* __restrict__ ref,
	RES* __restrict__ res,
	int2_t size,
	int2_t res_size,
	esize_t ref_slices,
	esize_t batch_size)
{
	esize_t slice_num = blockIdx.x / res_size.y;
	esize_t res_y = blockIdx.x % res_size.y;

	ref += slice_num * size.area();
	pics += slice_num * size.area();
	res += slice_num * res_size.area();

	T* smem = shared_memory_proxy<T>();
	T* res_line = smem;


	for (int i = threadIdx.x; i < res_size.x; i += blockDim.x)
	{
		res_line[i] = 0;
	}
	__syncthreads();


	int2_t res_r = (res_size - 1) / 2;
	
	int y_shift = res_y - (int)res_r.y;
	int y_begin = y_shift >= 0 ? 0 : -y_shift;


	int warp_idx = threadIdx.x / warpSize;
	int lane_idx = threadIdx.x % warpSize;
	int team_size = warpSize / stripe_size;
	int team_idx = lane_idx / team_size;
	int team_lane = lane_idx % team_size;

	//T sums[30];
	for (int s = 0; s < size.y - abs(y_shift); s += stripe_size)
	{
		
		for (int x_shift = -res_r.x + warp_idx; x_shift <= res_r.x; x_shift += blockDim.x / warpSize)
		{
			T sum = 0;

			int x_end = x_shift < 0 ? size.x : size.x - x_shift;
			int x_begin = x_shift < 0 ? -x_shift : 0;
			
			int y = s + y_begin + team_idx;
			int y_shifted = y + y_shift;
			if(y < size.y && y_shifted < size.y)
				for (int x = x_begin + team_lane; x < x_end; x += team_size)
				{
					int x_shifted = x + x_shift;
					
					//if(blockIdx.x < 1)
						//printf("%d %d %d %d %d [%d %d] [%d %d] [%d %d] %d\n", blockIdx.x, s, warp_idx, lane_idx, y_begin, x_shift, y_shift, x, y, x_shifted, y_shifted, x_end);
					sum += pics[y_shifted * size.x + x_shifted] * ref[y * size.x + x];
				}
			//printf("%d %d %d %d %d [%d %d] %f %d\n", blockIdx.x, s, warp_idx, lane_idx, y_begin, x_shift, y_shift, sum, x_shift + r.x);

			for (int offset = warpSize / 2; offset > 0; offset /= 2)
				sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);

			if(lane_idx == 0)
				*(res_line + x_shift + res_r.x) += sum;

		}
		
	}

	__syncthreads();

	for (int i = threadIdx.x; i < res_size.x; i += blockDim.x)
	{
		res[res_size.x * res_y + i] = res_line[i];
	}
}

template<typename T, typename RES>
void run_cross_corr_opt(
	const T* pics,
	const T* ref,
	RES* res,
	size2_t size,
	size2_t res_size,
	esize_t ref_slices,
	esize_t batch_size)
{
	esize_t block_dim = 256;
	esize_t grid_size = res_size.y * ref_slices;
	esize_t shared_mem_size = res_size.x * sizeof(T) * 2;
	cross_corr_opt<T, RES> <<<grid_size, block_dim, shared_mem_size >>> (pics, ref, res, { (int)size.x, (int)size.y }, { (int)res_size.x, (int)res_size.y }, ref_slices, batch_size);
}

template void run_cross_corr_opt<double, double>(
	const double*,
	const double*,
	double* res,
	size2_t size,
	size2_t res_size,
	esize_t,
	esize_t);

template void run_cross_corr_opt<float, float>(
	const float*,
	const float*,
	float* res,
	size2_t size,
	size2_t res_size,
	esize_t,
	esize_t);





//*******************************************************************************************************************************************************
constexpr int stripe_esize_tr = 32;

template<typename T, typename RES>
__global__ void cross_corr_opt_tr(
	const T* __restrict__ pics,
	const T* __restrict__ ref,
	RES* __restrict__ res,
	int2_t size,
	int2_t res_size,
	esize_t ref_slices,
	esize_t batch_size)
{
	esize_t slice_num = blockIdx.x / res_size.x;
	esize_t res_x = blockIdx.x % res_size.x;

	ref += slice_num * size.area();
	pics += slice_num * size.area();
	res += slice_num * res_size.area();

	T* smem = shared_memory_proxy<T>();
	T* res_line = smem;


	for (int i = threadIdx.x; i < res_size.y; i += blockDim.x)
	{
		res_line[i] = 0;
	}
	__syncthreads();


	int2_t res_r = (res_size - 1) / 2;

	int x_shift = res_x - (int)res_r.x;
	int x_begin = x_shift >= 0 ? 0 : -x_shift;


	int warp_idx = threadIdx.x / warpSize;
	int lane_idx = threadIdx.x % warpSize;
	int team_size = warpSize / stripe_esize_tr;
	int team_idx = lane_idx / team_size;
	//int team_lane = lane_idx % team_size;

	constexpr int k = 1;
	//T sums[30];
	for (int s = 0; s < size.x - abs(x_shift); s += stripe_esize_tr)
	{

		for (int y_shift = -res_r.y + warp_idx*k; y_shift <= res_r.y; y_shift += blockDim.x / warpSize * k)
		{
			
			T sum[k];
			//T cach[k];
			#pragma unroll
			for (int i = 0; i < k; ++i)
				sum[i] = 0;

			int y_end = y_shift < 0 ? size.y : size.y - y_shift;
			int y_begin = y_shift < 0 ? -y_shift : 0;

			int x = s + x_begin + team_idx;
			int x_shifted = x + x_shift;

			if (x < size.x && x_shifted < size.x)
			{

				for (int y = y_begin; y < y_end; y += 1)
				{
					int y_shifted = y + y_shift;

					//if(blockIdx.x < 1)
						//printf("%d %d %d %d %d [%d %d] [%d %d] [%d %d] %d\n", blockIdx.x, s, warp_idx, lane_idx, y_begin, x_shift, y_shift, x, y, x_shifted, y_shifted, x_end);
					#pragma unroll
					for (int i = 0; i < k; ++i)
						sum[i] += pics[y_shifted * size.x + x_shifted + i] * ref[y * size.x + x];

					/*++y_shifted;
					cach[2] = pics[y_shifted * size.x + x_shifted];
					//sum[0] += pics[y_shifted * size.x + x_shifted] * ref[y * size.x + x];
					//sum[1] += pics[y_shifted * size.x + x_shifted + 1] * ref[y * size.x + x];
					sum[0] += cach[0] * ref[y * size.x + x];
					sum[1] += cach[1] * ref[y * size.x + x];
					sum[2] += cach[2] * ref[y * size.x + x];

					++y_shifted;
					cach[0] = pics[y_shifted * size.x + x_shifted];
					sum[0] += cach[1] * ref[y * size.x + x + 1];
					sum[1] += cach[2] * ref[y * size.x + x + 1];
					sum[2] += cach[0] * ref[y * size.x + x + 1];

					++y_shifted;
					cach[1] = pics[y_shifted * size.x + x_shifted];
					sum[0] += cach[2] * ref[y * size.x + x + 2];
					sum[1] += cach[0] * ref[y * size.x + x + 2];
					sum[2] += cach[1] * ref[y * size.x + x + 2];*/

				}
			}
			//printf("%d %d %d %d %d [%d %d] %f %d\n", blockIdx.x, s, warp_idx, lane_idx, y_begin, x_shift, y_shift, sum, x_shift + r.x);

			for (int offset = warpSize / 2; offset > 0; offset /= 2)
				sum[0] += __shfl_down_sync(0xFFFFFFFF, sum[0], offset);

			if (lane_idx == 0)
			{
				#pragma unroll
				for (int i = 0; i < k; ++i)
					*(res_line + y_shift + res_r.x + i) += sum[i];
			}

		}

	}

	__syncthreads();

	for (int i = threadIdx.x; i < res_size.y; i += blockDim.x)
	{
		res[res_size.x * i + res_x] = res_line[i];
	}
}

template<typename T, typename RES>
void run_cross_corr_opt_tr(
	const T* pics,
	const T* ref,
	RES* res,
	size2_t size,
	size2_t res_size,
	esize_t ref_slices,
	esize_t batch_size)
{
	esize_t block_dim = 256;
	esize_t grid_size = res_size.y * ref_slices;
	esize_t shared_mem_size = res_size.x * sizeof(T) * 2;
	cross_corr_opt_tr<T, RES> << <grid_size, block_dim, shared_mem_size >> > (pics, ref, res, { (int)size.x, (int)size.y }, { (int)res_size.x, (int)res_size.y }, ref_slices, batch_size);
}

template void run_cross_corr_opt_tr<double, double>(
	const double*,
	const double*,
	double* res,
	size2_t size,
	size2_t res_size,
	esize_t,
	esize_t);

template void run_cross_corr_opt_tr<float, float>(
	const float*,
	const float*,
	float* res,
	size2_t size,
	size2_t res_size,
	esize_t,
	esize_t);

}