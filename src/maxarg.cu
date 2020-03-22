
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
__global__ void maxarg_reduce(const T* data, data_index<T> * maxes, size_t size)
{
	extern __shared__ data_index<T> sdata[];

	size_t tid = threadIdx.x;
	
	//number of blocks we need to process one picture
	size_t one_pic_blocks = div_up(size, blockDim.x);
	size_t pic_num = blockIdx.x / one_pic_blocks;
	size_t pic_block = blockIdx.x % one_pic_blocks;

	//if this is the last block that processes one picture(chunk)
	//and this thread would process sth out of the picture
	if (blockIdx.x % one_pic_blocks == one_pic_blocks-1 && threadIdx.x >= size % blockDim.x)
		return;

	size_t i = pic_num * size + pic_block * blockDim.x + threadIdx.x;
	sdata[tid].data = data[i];
	sdata[tid].index = i;
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

template<typename T, int s>
__global__ void extract_neighbors(const T* data, const vec2<size_t> * max_i, T* neighbors, size_t cols, size_t rows, size_t batch_size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (idx >= batch_size)
		return;
	
	size_t r = (s - 1) / 2;

	auto max = max_i[idx];

	for (int i = 0; i < s; ++i)
	{
		for (int j = 0; j < s; ++j)
		{
			size_t from_x = max.x - r + i;
			size_t from_y = max.y - r + j;

			neighbors[idx * s * s + j * s + i] = data[idx * cols * rows + from_y * cols + from_x];
		}
	}	
}

template<typename T>
void run_maxarg_reduce(const T* data, data_index<T>* maxes, size_t size, size_t block_size, size_t batch_size)
{	
	size_t one_pic_blocks = div_up(size, block_size);
	size_t grid_size = one_pic_blocks * batch_size;
	maxarg_reduce<T> <<<grid_size, block_size, block_size * sizeof(data_index<T>)>>> (data, maxes, size);
}

template<typename T, int s>
void run_extract_neighbors(const T* data, const vec2<size_t>* max_i, T* neighbors, size_t cols, size_t rows, size_t batch_size)
{
	size_t block_size = 128;
	size_t grid_size = div_up(batch_size, block_size);
	extract_neighbors<T,s> <<<grid_size, block_size>>> (data, max_i, neighbors, cols, rows, batch_size);
}


template void run_maxarg_reduce<double>(const double* data, data_index<double>* maxes, size_t size, size_t block_size, size_t batch_size);

template void run_extract_neighbors<double, 3>(const double* data, const vec2<size_t>* max_i, double* neighbors, size_t cols, size_t rows, size_t batch_size);

}