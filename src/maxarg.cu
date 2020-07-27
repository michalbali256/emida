
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "device_helpers.hpp"
#include "kernels.cuh"

#include <type_traits>

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

__device__ __inline__ uint64_t dataindex_as_uint64(data_index<float> val)
{
	static_assert(sizeof(data_index<float>) == sizeof(uint64_t), "Wrong data_index<float> length.");
	return *reinterpret_cast<uint64_t *>(&val);
}


__device__ __inline__ data_index<float> uint64_as_dataindex(uint64_t val)
{
	static_assert(sizeof(data_index<float>) == sizeof(uint64_t), "Wrong data_index<float> length.");
	return *reinterpret_cast<data_index<float>*>(&val);
}

template<typename T>
__device__ __inline__ void update_res(const data_index<T>& res, data_index<T>* __restrict__ maxes, esize_t pic_num);

template<>
__device__ __inline__ void update_res<double>(const data_index<double>& res, data_index<double>* __restrict__ maxes, esize_t pic_num)
{
	maxes[blockIdx.x] = res;
}

template<>
__device__ __inline__ void update_res<float>(const data_index<float>& res, data_index<float>* __restrict__ maxes, esize_t pic_num)
{
	//maxes[blockIdx.x] = res;

	uint64_t* address_as_ull = (uint64_t *)maxes + pic_num;
	uint64_t old = *address_as_ull;
	uint64_t assumed;

	do {
		assumed = old;
		if(uint64_as_dataindex(assumed).data < res.data)
			old = atomicCAS(address_as_ull, assumed, dataindex_as_uint64(res));

		// Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
	} while (assumed != old);
}


constexpr int N = 10;

//        size
//      +-------+-------+-------+
//      +--+--+--+      +--+--+--+
//              +--+--+--+
//blockDim2  2 2  2 2  2  2  2  2  
//block: 1  2  3  4  5  6 7  8  9
template<typename T>
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
	
	esize_t slice_end = (pic_num + 1) * slice_size.area();
	esize_t i = pic_num * slice_size.area() + slice_tid;
	
	data_index<T> val;

	if (i >= slice_end)
	{
		val.data = 0;
		val.index = slice_tid;
	}
	else
	{
		val.data = data[i];
		val.index = slice_tid;
		for (esize_t n = 1; n < N; ++n)
		{
			i += one_pic_blocks * blockDim.x;
			slice_tid += one_pic_blocks * blockDim.x;

			if (i < slice_end)
			{
				if (data[i] > val.data)
				{
					val.data = data[i];
					val.index = slice_tid;
				}
			}
		}
	}
	
	__syncthreads();

	data_index<T> res = blockReduceSum(val);
	
	if (tid == 0)
	{
		if(one_pic_blocks == 1)
			maxes[blockIdx.x] = res;
		else
			update_res<T>(res, maxes, pic_num);
	}
}

template<typename T>
__global__ void maxarg_reduce2(const data_index<T>* __restrict__ maxes_in, data_index<T>* __restrict__ maxes_out, esize_t one_pic_blocks, size2_t pic_size)
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
		//esize_t max_i = res.index - blockIdx.x * pic_size.area();

		maxes_out[blockIdx.x] = res;
	}
}

template<typename T>
void run_maxarg_reduce(const T* data, data_index<T>* maxes_red, data_index<T>* maxarg, size2_t size, esize_t block_size, esize_t batch_size)
{	
	esize_t one_pic_blocks = div_up(size.area(), block_size*N);

	esize_t grid_size = one_pic_blocks * batch_size;
	if(one_pic_blocks == 1 || std::is_same<T, float>::value)
		maxarg_reduce<T> <<<grid_size, block_size, block_size * sizeof(data_index<T>)>>> (data, maxarg, size);
	else
	{
		maxarg_reduce<T> <<<grid_size, block_size, block_size * sizeof(data_index<T>) >>> (data, maxes_red, size);
		maxarg_reduce2<T> <<<batch_size, 1024, 1024 * sizeof(data_index<T>)>>> (maxes_red, maxarg, one_pic_blocks, size);
	}
}

template void run_maxarg_reduce<double>(const double* data, data_index<double>* maxes, data_index<double>* maxarg, size2_t size, esize_t block_size, esize_t batch_size);
template void run_maxarg_reduce<float>(const float* data, data_index<float>* maxes, data_index<float>* maxarg, size2_t size, esize_t block_size, esize_t batch_size);

}