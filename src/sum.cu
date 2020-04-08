
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

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
	extern __shared__ T sdata[];

	size_t tid = threadIdx.x;
	
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
		sdata[tid] = 0;
	}
	else
		sdata[tid] = data[i];
	
	if (tid == 0 && pic_block == 0)
		maxes[pic_num] = 0;

	__syncthreads();

	for (size_t s = blockDim.x / 2; s > 0; s >>= 1)
	{
		if (tid < s)
		{
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}
	
	if (tid == 0)
		atomicAdd(&maxes[pic_num], sdata[0]);
}

template<typename T>
void run_sum(const T* data, T * maxes, size_t size, size_t batch_size)
{	
	size_t block_size = 1024;
	size_t one_pic_blocks = div_up(size, block_size);
	size_t grid_size = one_pic_blocks * batch_size;
	sum<T> <<<grid_size, block_size, block_size * sizeof(T)>>> (data, maxes, size);
}

template void run_sum<double>(const double* data, double * maxes, size_t size, size_t batch_size);

}