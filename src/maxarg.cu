
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "device_helpers.hpp"
#include "kernels.cuh"

namespace emida
{
//        size
//      +-------+-------+-------+
//      +--+--+--+      +--+--+--+
//              +--+--+--+
//blockDim2  2 2  2 2  2  2  2  2  
//block: 1  2  3  4  5  6 7  8  9
template<typename T, class pos_policy>
__global__ void maxarg_reduce(const T* __restrict__ data, data_index<T> * __restrict__ maxes, size2_t slice_size)
{
	data_index<T> * sdata = shared_memory_proxy<data_index<T>>();

	esize_t tid = threadIdx.x;
	
	//number of blocks we need to process one picture
	esize_t one_pic_blocks = div_up(slice_size.area(), blockDim.x);
	one_pic_blocks = div_up(one_pic_blocks, 2);

	esize_t pic_num = blockIdx.x / one_pic_blocks;
	esize_t pic_block = blockIdx.x % one_pic_blocks;

	//if this is the last block that processes one picture(chunk)
	//and this thread would process sth out of the picture

	esize_t slice_tid = pic_block * blockDim.x + threadIdx.x;
	size2_t slice_pos = { slice_tid % slice_size.x, slice_tid / slice_size.x };

	esize_t i = pic_num * slice_size.area() + slice_pos.pos(slice_size.x);
	if (slice_pos.x >= slice_size.x || slice_pos.y >= slice_size.y)
	{
		sdata[tid].data = 0;
		sdata[tid].index = i;
	}
	else
	{
		sdata[tid].data = data[pos_policy::pos(i, pic_num, slice_pos, slice_size)];
		sdata[tid].index = i;

		slice_tid += one_pic_blocks * blockDim.x;
		slice_pos = { slice_tid % slice_size.x, slice_tid / slice_size.x };
		i = pic_num * slice_size.area() + slice_pos.pos(slice_size.x);

		if (slice_pos.x < slice_size.x && slice_pos.y < slice_size.y)
		{
			esize_t sh_i = pos_policy::pos(i, pic_num, slice_pos, slice_size);
			if (data[sh_i] > sdata[tid].data)
			{
				sdata[tid].data = data[sh_i];
				sdata[tid].index = i;
			}
		}
	}


	
	__syncthreads();

	for (esize_t s = blockDim.x / 2; s > 0; s >>= 1)
	{
		if (tid < s)
		{
			if (sdata[tid + s].data > sdata[tid].data)
			{
				sdata[tid] = sdata[tid + s];
			}
		}
		__syncthreads();
	}
	
	if (tid == 0) maxes[blockIdx.x] = sdata[0];
}

template<typename T>
__global__ void maxarg_reduce2(const data_index<T>* __restrict__ maxes_in, size2_t * __restrict__ maxes_out, esize_t one_pic_blocks, size2_t pic_size)
{
	data_index<T>* sdata = shared_memory_proxy<data_index<T>>();

	esize_t tid = threadIdx.x;

	esize_t i = blockIdx.x * one_pic_blocks + threadIdx.x;
	if (threadIdx.x >= one_pic_blocks)
	{
		sdata[tid].data = 0;
		sdata[tid].index = i;
	}
	else
	{
		sdata[tid] = maxes_in[i];
	}

	__syncthreads();

	for (esize_t s = blockDim.x / 2; s > 0; s >>= 1)
	{
		if (tid < s)
		{
			if (sdata[tid + s].data > sdata[tid].data)
			{
				sdata[tid] = sdata[tid + s];
			}
		}
		__syncthreads();
	}

	if (tid == 0)
	{
		esize_t max_i = sdata[0].index - blockIdx.x * pic_size.area();
		maxes_out[blockIdx.x].x = max_i % pic_size.x;
		maxes_out[blockIdx.x].y = max_i / pic_size.x;
	}
}

template<typename T, class pos_policy>
void run_maxarg_reduce(const T* data, data_index<T>* maxes_red, size2_t * maxarg, size2_t size, esize_t block_size, esize_t batch_size)
{	
	esize_t one_pic_blocks = div_up(size.area(), block_size);

	esize_t grid_size = div_up(one_pic_blocks, 2) * batch_size;
	maxarg_reduce<T, pos_policy> <<<grid_size, block_size, block_size * sizeof(data_index<T>)>>> (data, maxes_red, size);
	maxarg_reduce2<T> <<<batch_size, 1024, 1024 * sizeof(data_index<T>)>>> (maxes_red, maxarg, div_up(one_pic_blocks,2), size);
}

template void run_maxarg_reduce<double, cross_res_pos_policy_id>(const double* data, data_index<double>* maxes, size2_t* maxarg, size2_t size, esize_t block_size, esize_t batch_size);
template void run_maxarg_reduce<double, cross_res_pos_policy_fft>(const double* data, data_index<double>* maxes, size2_t* maxarg, size2_t size, esize_t block_size, esize_t batch_size);
template void run_maxarg_reduce<float, cross_res_pos_policy_id>(const float* data, data_index<float>* maxes, size2_t* maxarg, size2_t size, esize_t block_size, esize_t batch_size);
template void run_maxarg_reduce<float, cross_res_pos_policy_fft>(const float* data, data_index<float>* maxes, size2_t* maxarg, size2_t size, esize_t block_size, esize_t batch_size);

}