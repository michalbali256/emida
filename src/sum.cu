
#include "cuda.h"
#include "cuda_runtime.h"

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
__global__ void sum(const T* data, T * maxes, size_t size)
{
	T* sdata = shared_memory_proxy<T>();
	
	//number of blocks we need to process one picture
	size_t one_pic_blocks = div_up(size, blockDim.x);
	size_t pic_num = blockIdx.x / one_pic_blocks;
	size_t pic_block = blockIdx.x % one_pic_blocks;

	//if this is the last block that processes one picture(chunk)
	//and this thread would process sth out of the picture, just load zero

	size_t i = pic_num * size + pic_block * blockDim.x + threadIdx.x;
	if (blockIdx.x % one_pic_blocks == one_pic_blocks - 1
		&& size % blockDim.x != 0
		&& threadIdx.x >= size % blockDim.x)
	{
		sdata[threadIdx.x] = 0;
	}
	else
		sdata[threadIdx.x] = data[i];
	
	if (threadIdx.x == 0 && pic_block == 0)
		maxes[pic_num] = 0;

	__syncthreads();

	for (size_t s = blockDim.x / 2; s > 0; s >>= 1)
	{
		if (threadIdx.x < s)
		{
			sdata[threadIdx.x] += sdata[threadIdx.x + s];
		}
		__syncthreads();
	}
	
	if (threadIdx.x == 0)
		atomicAdd(&maxes[pic_num], sdata[0]);
}

template<typename T>
void run_sum(const T* data, T * sums, size_t size, size_t batch_size)
{	
	size_t block_size = 1024;
	size_t one_pic_blocks = div_up(size, block_size);
	size_t grid_size = one_pic_blocks * batch_size;
	sum<T> <<<grid_size, block_size, block_size * sizeof(T)>>> (data, sums, size);
}

template void run_sum<double>(const double* data, double * sums, size_t size, size_t batch_size);


template<typename T, typename RES>
__global__ void sum(
	const T* data,
	RES* sums,
	const size2_t* begins,
	size2_t src_size,
	size2_t slice_size,
	size_t begins_size)
{
	RES* sdata = shared_memory_proxy<RES>();

	//number of blocks we need to process one slice
	size_t one_slice_blocks = div_up(slice_size.area(), blockDim.x);
	size_t slice_num = blockIdx.x / one_slice_blocks;
	size_t slice_block = blockIdx.x % one_slice_blocks;
	
	//if this is the last block that processes one picture(chunk)
	//and this thread would process sth out of the picture, just load zero
	
	if (blockIdx.x % one_slice_blocks == one_slice_blocks - 1
		&& slice_size.area() % blockDim.x != 0
		&& threadIdx.x >= slice_size.area() % blockDim.x)
	{
		sdata[threadIdx.x] = 0;
	}
	else
	{
		size_t begins_num = slice_num % begins_size;
		size_t pic_num = slice_num / begins_size;
		size_t slice_i = slice_block * blockDim.x + threadIdx.x;
		size2_t slice_pos = { slice_i % slice_size.x, slice_i / slice_size.x };
		size2_t src_pos = begins[begins_num] + slice_pos;

		sdata[threadIdx.x] = data[pic_num * src_size.area() + src_pos.pos(src_size.x)];
	}
	if (threadIdx.x == 0 && slice_block == 0)
		sums[slice_num] = 0;

	__syncthreads();

	for (size_t s = blockDim.x / 2; s > 0; s >>= 1)
	{
		if (threadIdx.x < s)
		{
			sdata[threadIdx.x] += sdata[threadIdx.x + s];
		}
		__syncthreads();
	}

	if (threadIdx.x == 0)
		atomicAdd(&sums[slice_num], sdata[0]);
}

template<typename T, typename RES>
void run_sum(const T* data, RES* sums, const size2_t * begins, size2_t src_size, size2_t slice_size, size_t begins_size, size_t batch_size)
{
	size_t block_size = 1024;
	size_t one_pic_blocks = div_up(slice_size.area(), block_size);
	size_t grid_size = one_pic_blocks * begins_size * batch_size;
	sum<T, RES> <<<grid_size, block_size, block_size * sizeof(RES)>>> (data, sums, begins, src_size, slice_size, begins_size);
}

template void run_sum<double, double>(const double* data,
	double* sums,
	const size2_t* begins,
	size2_t src_size,
	size2_t slice_size,
	size_t begins_size,
	size_t batch_size);
//template void run_sum<uint16_t, uint32_t>(const uint16_t* data, uint32_t* sums, const size2_t* begins, size2_t src_size, size2_t slice_size, size_t batch_size);
template void run_sum<uint16_t, double>(const uint16_t* data,
	double* sums,
	const size2_t* begins,
	size2_t src_size,
	size2_t slice_size,
	size_t begins_size,
	size_t batch_size);
template void run_sum<uint16_t, float>(const uint16_t* data,
	float* sums,
	const size2_t* begins,
	size2_t src_size,
	size2_t slice_size,
	size_t begins_size,
	size_t batch_size);

}