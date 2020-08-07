#include "cuda.h"
#include "cuda_runtime.h"

#include "kernels.cuh"

namespace emida
{

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
	esize_t slice_tid = tid % slice_size.area();
	esize_t slice_num = tid / slice_size.area();
	size2_t slice_pos = { slice_tid % slice_size.x, slice_tid / slice_size.x };
	esize_t begins_num = slice_num % begins_size;
	esize_t pic_num = slice_num / begins_size;
	if (slice_num >= begins_size * batch_size)
		return;

	size2_t pic_pos = begins[begins_num] + slice_pos;

	OUT pixel = pic[pic_num * src_size.area() + pic_pos.pos(src_size.x)];
	//subtract mean of the picture
	pixel -= (OUT)sums[slice_num] / slice_size.area();
	//apply hanning filter and convert to OUT (float or double)
	pixel = (OUT)pixel * hanning_x[slice_pos.x] * hanning_y[slice_pos.y];
	slices[slice_num * out_size.area() + slice_pos.pos(out_size.x)] = pixel;
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
	esize_t grid_size(div_up(slice_size.area() * batch_size * begins_size, block_size));
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
