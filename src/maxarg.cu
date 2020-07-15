
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "device_helpers.hpp"
#include "kernels.cuh"

namespace emida
{

constexpr int warp_size = 32;

template<typename T>
__inline__ __device__ data_index<T> warpReduceSum(data_index<T> val)
{
	#pragma unroll
	for (int offset = warp_size / 2; offset > 0; offset /= 2)
	{
		T next_data = __shfl_down_sync(0xFFFFFFFF, val.data, offset);
		esize_t next_index = __shfl_down_sync(0xFFFFFFFF, val.index, offset);
		if (next_data > val.data)
		{
			val.data = next_data;
			val.index = next_index;
		}
	}
	return val;
}


template<typename T>
__inline__ __device__ T blockReduceSum(T val) {

	T* shared = shared_memory_proxy<T>();
	int lane = threadIdx.x % warpSize;
	int warp_id = threadIdx.x / warpSize;

	val = warpReduceSum(val);     // Each warp performs partial reduction

	if (lane == 0) shared[warp_id] = val; // Write reduced value to shared memory

	__syncthreads();              // Wait for all partial reductions

	if (warp_id == 0)
	{
		val = shared[lane];

		val = warpReduceSum(val);
	}

	return val;
}

constexpr int N = 3;

//        size
//      +-------+-------+-------+
//      +--+--+--+      +--+--+--+
//              +--+--+--+
//blockDim2  2 2  2 2  2  2  2  2  
//block: 1  2  3  4  5  6 7  8  9
template<typename T, class pos_policy>
__global__ void maxarg_reduce(const T* __restrict__ data, data_index<T> * __restrict__ maxes, size2_t slice_size)
{
	esize_t tid = threadIdx.x;
	
	//number of blocks we need to process one picture
	esize_t one_pic_blocks = div_up(slice_size.area(), blockDim.x * N);

	esize_t pic_num = blockIdx.x / one_pic_blocks;
	esize_t pic_block = blockIdx.x % one_pic_blocks;

	//if this is the last block that processes one picture(chunk)
	//and this thread would process sth out of the picture

	esize_t slice_tid = pic_block * blockDim.x + threadIdx.x;
	size2_t slice_pos = { slice_tid % slice_size.x, slice_tid / slice_size.x };

	

	esize_t i = pic_num * slice_size.area() + slice_pos.pos(slice_size.x);
	
	data_index<T> val;

	if (slice_pos.x >= slice_size.x || slice_pos.y >= slice_size.y)
	{
		val.data = 0;
		val.index = i;
	}
	else
	{
		val.data = data[pos_policy::pos(i, pic_num, slice_pos, slice_size)];
		val.index = i;
		for (esize_t n = 1; n < N; ++n)
		{
			slice_tid += one_pic_blocks * blockDim.x;
			slice_pos = { slice_tid % slice_size.x, slice_tid / slice_size.x };
			i = pic_num * slice_size.area() + slice_pos.pos(slice_size.x);

			if (slice_pos.x < slice_size.x && slice_pos.y < slice_size.y)
			{
				esize_t sh_i = pos_policy::pos(i, pic_num, slice_pos, slice_size);
				if (data[sh_i] > val.data)
				{
					val.data = data[sh_i];
					val.index = i;
				}
			}
		}
	}
	
	__syncthreads();

	data_index<T> res = blockReduceSum(val);
	
	if (tid == 0) maxes[blockIdx.x] = res;
}

template<typename T>
__global__ void maxarg_reduce2(const data_index<T>* __restrict__ maxes_in, size2_t * __restrict__ maxes_out, esize_t one_pic_blocks, size2_t pic_size)
{

	esize_t tid = threadIdx.x;
	data_index<T> val;

	esize_t i = blockIdx.x * one_pic_blocks + threadIdx.x;
	if (threadIdx.x >= one_pic_blocks)
	{
		val.data = 0;
		val.index = i;
	}
	else
	{
		val = maxes_in[i];
	}

	__syncthreads();

	data_index<T> res = blockReduceSum(val);

	if (tid == 0)
	{
		esize_t max_i = res.index - blockIdx.x * pic_size.area();
		maxes_out[blockIdx.x].x = max_i % pic_size.x;
		maxes_out[blockIdx.x].y = max_i / pic_size.x;
	}
}

template<typename T, class pos_policy>
void run_maxarg_reduce(const T* data, data_index<T>* maxes_red, size2_t * maxarg, size2_t size, esize_t block_size, esize_t batch_size)
{	
	esize_t one_pic_blocks = div_up(size.area(), block_size*N);

	esize_t grid_size = one_pic_blocks * batch_size;
	maxarg_reduce<T, pos_policy> <<<grid_size, block_size, block_size * sizeof(data_index<T>)>>> (data, maxes_red, size);
	maxarg_reduce2<T> <<<batch_size, 1024, 1024 * sizeof(data_index<T>)>>> (maxes_red, maxarg, one_pic_blocks, size);
}

template void run_maxarg_reduce<double, cross_res_pos_policy_id>(const double* data, data_index<double>* maxes, size2_t* maxarg, size2_t size, esize_t block_size, esize_t batch_size);
template void run_maxarg_reduce<double, cross_res_pos_policy_fft>(const double* data, data_index<double>* maxes, size2_t* maxarg, size2_t size, esize_t block_size, esize_t batch_size);
template void run_maxarg_reduce<float, cross_res_pos_policy_id>(const float* data, data_index<float>* maxes, size2_t* maxarg, size2_t size, esize_t block_size, esize_t batch_size);
template void run_maxarg_reduce<float, cross_res_pos_policy_fft>(const float* data, data_index<float>* maxes, size2_t* maxarg, size2_t size, esize_t block_size, esize_t batch_size);

}