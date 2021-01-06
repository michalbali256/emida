
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
__global__ void sum(const T* data, T * maxes, esize_t size)
{
	T* sdata = shared_memory_proxy<T>();
	
	//number of blocks we need to process one picture
	esize_t one_pic_blocks = div_up(size, blockDim.x);
	esize_t pic_num = blockIdx.x / one_pic_blocks;
	esize_t pic_block = blockIdx.x % one_pic_blocks;

	//if this is the last block that processes one picture(chunk)
	//and this thread would process sth out of the picture, just load zero

	esize_t i = pic_num * size + pic_block * blockDim.x + threadIdx.x;
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

	for (esize_t s = blockDim.x / 2; s > 0; s >>= 1)
	{
		if (threadIdx.x < s)
		{
			sdata[threadIdx.x] += sdata[threadIdx.x + s];
		}
		__syncthreads();
	}
	
	if (threadIdx.x == 0)
		atomicAdd((T*) &maxes[pic_num], (T) sdata[0]);
}

template<typename T>
void run_sum(const T* data, T * sums, esize_t size, esize_t batch_size)
{	
	esize_t block_size = 1024;
	esize_t one_pic_blocks = div_up(size, block_size);
	esize_t grid_size = one_pic_blocks * batch_size;
	sum<T> <<<grid_size, block_size, block_size * sizeof(T)>>> (data, sums, size);
}

template void run_sum<double>(const double* data, double * sums, esize_t size, esize_t batch_size);

constexpr int warp_size = 32;

template<typename T>
__inline__ __device__ T warpReduceSum(T val)
{	
	#pragma unroll
	for (int offset = warp_size / 2; offset > 0; offset /= 2)
		val += __shfl_down_sync(0xFFFFFFFF, val, offset);
	return val;
}


template<typename T>
__inline__ __device__ T blockReduceSum(T val)
{
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
	} //Final reduce within first warp

	return val;
}

constexpr int N = 10;

template<typename T, typename RES>
__global__ void sum(
	const T* data,
	RES* sums,
	const size2_t* begins,
	size2_t src_size,
	size2_t slice_size,
	esize_t begins_size)
{
	//RES* sdata = shared_memory_proxy<RES>();

	//number of blocks we need to process one slice
	esize_t one_slice_blocks = div_up(slice_size.area(), blockDim.x * N);
	esize_t slice_num = blockIdx.x / one_slice_blocks;
	esize_t slice_block = blockIdx.x % one_slice_blocks;
	
	
	if (threadIdx.x == 0 && slice_block == 0)
		sums[slice_num] = 0;
	__syncthreads();

	esize_t begins_num = slice_num % begins_size;
	esize_t pic_num = slice_num / begins_size;
	RES val = 0;

	for (esize_t n = 0; n < N; ++n)
	{
		esize_t slice_i = one_slice_blocks * blockDim.x * n + slice_block * blockDim.x + threadIdx.x;
		size2_t slice_pos = { slice_i % slice_size.x, slice_i / slice_size.x };

		if (slice_pos.x < slice_size.x && slice_pos.y < slice_size.y)
		{
			size2_t src_pos = begins[begins_num] + slice_pos;

			val += (RES)data[pic_num * src_size.area() + src_pos.pos(src_size.x)];
		}
	}


	RES res = blockReduceSum(val);

	if (threadIdx.x == 0)
	{
		if (one_slice_blocks == 1)
		{
			sums[slice_num] = res;
		}
		else
			atomicAdd((RES *)&sums[slice_num], (RES) res);
	}
}

template<typename T, typename RES>
void run_sum(const T* data, RES* sums, const size2_t * begins, size2_t src_size, size2_t slice_size, esize_t begins_size, esize_t batch_size)
{
	esize_t block_size = 1024;
	esize_t one_pic_blocks = div_up(slice_size.area(), block_size * N);
	esize_t grid_size = one_pic_blocks * begins_size * batch_size;
	sum<T, RES> <<<grid_size, block_size, warp_size * sizeof(RES)>>> (data, sums, begins, src_size, slice_size, begins_size);
}

template void run_sum<double, double>(const double* data,
	double* sums,
	const size2_t* begins,
	size2_t src_size,
	size2_t slice_size,
	esize_t begins_size,
	esize_t batch_size);

template void run_sum<double, uint32_t>(const double* data,
	uint32_t* sums,
	const size2_t* begins,
	size2_t src_size,
	size2_t slice_size,
	esize_t begins_size,
	esize_t batch_size);

template void run_sum<uint16_t, double>(const uint16_t* data,
	double* sums,
	const size2_t* begins,
	size2_t src_size,
	size2_t slice_size,
	esize_t begins_size,
	esize_t batch_size);
template void run_sum<uint16_t, float>(const uint16_t* data,
	float* sums,
	const size2_t* begins,
	size2_t src_size,
	size2_t slice_size,
	esize_t begins_size,
	esize_t batch_size);

template void run_sum<uint16_t, uint32_t>(const uint16_t* data,
	uint32_t* sums,
	const size2_t* begins,
	size2_t src_size,
	size2_t slice_size,
	esize_t begins_size,
	esize_t batch_size);

}