#include "cuda.h"
#include "cuda_runtime.h"

#include "kernels.cuh"

namespace emida
{

template<typename T>
__global__ void prepare_pics(T* __restrict pic, const T* hanning_x, const T* hanning_y, const T* sums, size2_t size, esize_t batch_size)
{
	esize_t whole_x = blockIdx.x * blockDim.x + threadIdx.x;
	esize_t pic_x = whole_x % size.x;
	esize_t pic_num = whole_x / size.x;
	esize_t y = blockIdx.y * blockDim.y + threadIdx.y;

	//Problem? many threads may end up nothing if number of rows and block_size.y are in poor combination
	if (pic_num >= batch_size || y >= size.y)
		return;

	T& pixel = pic[pic_num * size.area() + y * size.x + pic_x];
	//subtract mean of the picture
	pixel -= sums[pic_num] / size.area();
	//apply hanning filter
	pixel *= hanning_x[pic_x] * hanning_y[y];
}

template<typename T>
void run_prepare_pics(T* pic, const T* hanning_x, const T* hanning_y, const T * sums, size2_t size, esize_t batch_size)
{	
	dim3 block_size(16, 16);
	dim3 grid_size(div_up(size.x * batch_size, block_size.x), div_up(size.y, block_size.y));
	prepare_pics<T> <<<grid_size, block_size>>> (pic, hanning_x, hanning_y, sums, size, batch_size);
}


template void run_prepare_pics<double>(double * pic, const double * hanning_x,
	const double* hanning_y, const double * sums, size2_t size, esize_t batch_size);



template<typename IN, typename OUT, typename S = OUT>
__global__ void prepare_pics(
	const IN* __restrict__ pic,
	OUT* __restrict__ slices,
	const OUT* __restrict__ hanning_x,
	const OUT* __restrict__ hanning_y,
	const S* __restrict__ sums,
	const size2_t* __restrict__ begins,
	size2_t src_size,
	size2_t slice_size,
	size2_t out_size,
	esize_t begins_size,
	esize_t batch_size)
{
	esize_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	esize_t slice_tid = tid % out_size.area();
	esize_t slice_num = tid / out_size.area();
	size2_t slice_pos = { slice_tid % out_size.x, slice_tid / out_size.x };
	esize_t begins_num = slice_num % begins_size;
	esize_t pic_num = slice_num / begins_size;
	if (slice_num >= begins_size * batch_size)
		return;

	size2_t pic_pos = begins[begins_num] + slice_pos;

	if (slice_pos.x >= slice_size.x || slice_pos.y >= slice_size.y)
	{
		slices[tid] = 0;
		return;
	}

	OUT pixel = pic[pic_num * src_size.area() + pic_pos.pos(src_size.x)];
	//subtract mean of the picture
	pixel -= (OUT)sums[slice_num] / slice_size.area();
	//apply hanning filter and convert to OUT (float or double)
	pixel = (OUT)pixel * hanning_x[slice_pos.x] * hanning_y[slice_pos.y];
	slices[tid] = pixel;
}



template<typename IN, typename OUT, typename S>
void run_prepare_pics(
	const IN* pic,
	OUT* slices,
	const OUT* hanning_x,
	const OUT* hanning_y,
	const S* sums,
	const size2_t* begins,
	size2_t src_size,
	size2_t slice_size,
	size2_t out_size,
	esize_t begins_size,
	esize_t batch_size)
{
	esize_t block_size = 1024;
	esize_t grid_size(div_up(out_size.area() * batch_size * begins_size, block_size));
	prepare_pics<<<grid_size, block_size >>> (pic, slices, hanning_x, hanning_y, sums, begins, src_size, slice_size, out_size, begins_size, batch_size);
}

template void run_prepare_pics<uint16_t, double>(
	const uint16_t* pic,
	double* slices,
	const double* hanning_x,
	const double* hanning_y,
	const double* sums,
	const size2_t* begins,
	size2_t src_size,
	size2_t slice_size,
	size2_t out_size,
	esize_t begins_size,
	esize_t batch_size);

template void run_prepare_pics<uint16_t, double, uint32_t>(
	const uint16_t* pic,
	double* slices,
	const double* hanning_x,
	const double* hanning_y,
	const uint32_t* sums,
	const size2_t* begins,
	size2_t src_size,
	size2_t slice_size,
	size2_t out_size,
	esize_t begins_size,
	esize_t batch_size);

template void run_prepare_pics<uint16_t, float>(
	const uint16_t* pic,
	float* slices,
	const float* hanning_x,
	const float* hanning_y,
	const float* sums,
	const size2_t* begins,
	size2_t src_size,
	size2_t slice_size,
	size2_t out_size,
	esize_t begins_size,
	esize_t batch_size);

template void run_prepare_pics<uint16_t, float, uint32_t>(
	const uint16_t* pic,
	float* slices,
	const float* hanning_x,
	const float* hanning_y,
	const uint32_t* sums,
	const size2_t* begins,
	size2_t src_size,
	size2_t slice_size,
	size2_t out_size,
	esize_t begins_size,
	esize_t batch_size);


template void run_prepare_pics<double, double>(
	const double* pic,
	double* slices,
	const double* hanning_x,
	const double* hanning_y,
	const double* sums,
	const size2_t* begins,
	size2_t src_size,
	size2_t slice_size,
	size2_t out_size,
	esize_t begins_size,
	esize_t batch_size);

template void run_prepare_pics<double, double, uint32_t>(
	const double* pic,
	double* slices,
	const double* hanning_x,
	const double* hanning_y,
	const uint32_t* sums,
	const size2_t* begins,
	size2_t src_size,
	size2_t slice_size,
	size2_t out_size,
	esize_t begins_size,
	esize_t batch_size);

}
