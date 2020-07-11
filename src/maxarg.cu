
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
template<typename T>
__global__ void maxarg_reduce(const T* __restrict__ data, data_index<T> * __restrict__ maxes, size2_t slice_size)
{
	data_index<T> * sdata = shared_memory_proxy<data_index<T>>();

	size_t tid = threadIdx.x;
	
	//number of blocks we need to process one picture
	size_t one_pic_blocks = div_up(slice_size.area(), blockDim.x);
	size_t pic_num = blockIdx.x / one_pic_blocks;
	size_t pic_block = blockIdx.x % one_pic_blocks;

	//if this is the last block that processes one picture(chunk)
	//and this thread would process sth out of the picture

	size_t slice_tid = pic_block * blockDim.x + threadIdx.x;
	size2_t slice_pos = { slice_tid % slice_size.x, slice_tid / slice_size.x };

	size_t i = pic_num * slice_size.area() + slice_pos.pos(slice_size.x);
	if (slice_pos.x >= slice_size.x || slice_pos.y >= slice_size.y)
	{
		sdata[tid].data = 0;
		sdata[tid].index = i;
	}
	else
	{
		sdata[tid].data = data[i];
		sdata[tid].index = i;
	}
	
	__syncthreads();

	for (size_t s = blockDim.x / 2; s > 0; s >>= 1)
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
__global__ void maxarg_reduce2(const data_index<T>* __restrict__ maxes_in, size2_t * __restrict__ maxes_out, size_t one_pic_blocks, size2_t pic_size)
{
	data_index<T>* sdata = shared_memory_proxy<data_index<T>>();

	size_t tid = threadIdx.x;

	size_t i = blockIdx.x * one_pic_blocks + threadIdx.x;
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

	for (size_t s = blockDim.x / 2; s > 0; s >>= 1)
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
		size_t max_i = sdata[0].index - blockIdx.x * pic_size.area();
		maxes_out[blockIdx.x].x = max_i % pic_size.x;
		maxes_out[blockIdx.x].y = max_i / pic_size.x;
	}
}

template<typename T>
void run_maxarg_reduce(const T* data, data_index<T>* maxes_red, size2_t * maxarg, size2_t size, size_t block_size, size_t batch_size)
{	
	size_t one_pic_blocks = div_up(size.area(), block_size);
	size_t grid_size = one_pic_blocks * batch_size;
	maxarg_reduce<T> <<<grid_size, block_size, block_size * sizeof(data_index<T>)>>> (data, maxes_red, size);
	maxarg_reduce2<T> <<<batch_size, 1024, 1024 * sizeof(data_index<T>)>>> (maxes_red, maxarg, one_pic_blocks, size);
}

template void run_maxarg_reduce<double>(const double* data, data_index<double>* maxes, size2_t* maxarg, size2_t size, size_t block_size, size_t batch_size);
template void run_maxarg_reduce<float>(const float* data, data_index<float>* maxes, size2_t* maxarg, size2_t size, size_t block_size, size_t batch_size);

}