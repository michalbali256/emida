#include "cuda.h"
#include "cuda_runtime.h"
#include "cufft.h"

#include "kernels.cuh"

namespace emida
{
template<typename T>
__device__ T complex_mul(T a, T b);
template<>
__device__ cuComplex complex_mul<cuComplex>(cuComplex A, cuComplex B)
{
	return cuCmulf(A, B);
}
template<>
__device__ cuDoubleComplex complex_mul<cuDoubleComplex>(cuDoubleComplex A, cuDoubleComplex B)
{
	return cuCmul(A, B);
}

constexpr double PI = 3.14159265358979323846;

template<typename T>
__global__ void hadamard(
	T * __restrict__ pics,
	const T * __restrict__ ref,
	const T* __restrict__ shx,
	const T* __restrict__ shy,
	size2_t one_size,
	size_t ref_slices,
	size_t batch_size)
{
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	
	size_t slice_tid = i % one_size.area();
	size_t slice_num = i / one_size.area();
	size2_t slice_pos = { slice_tid % one_size.x, slice_tid / one_size.x };
	size_t ref_i = i % (one_size.area() * ref_slices);

	if (slice_num >= batch_size * ref_slices)
		return;
	
	T mul = { (pics[i].x * ref[ref_i].x + pics[i].y * ref[ref_i].y),
		(-pics[i].x * ref[ref_i].y + pics[i].y * ref[ref_i].x)};
	mul = complex_mul(mul, shx[slice_pos.x]);
	mul = complex_mul(mul, shy[slice_pos.y]);
	size_t num_elements = (one_size.x - 1) * 2 * one_size.y;
	mul = { mul.x / num_elements, mul.y / num_elements };
	pics[i] = mul;
}

template<typename T>
__global__ void hadamard(T* __restrict__ pics,
	const T* __restrict__ ref,
	size2_t one_size,
	size_t ref_slices,
	size_t batch_size)
{
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	size_t ref_i = i % (one_size.area() * ref_slices);

	if (i >= one_size.area() * ref_slices * batch_size)
		return;

	
	pics[i] = { (pics[i].x * ref[ref_i].x + pics[i].y * ref[ref_i].y),
		(-pics[i].x * ref[ref_i].y + pics[i].y * ref[ref_i].x) };
}

template<typename T>
void run_hadamard(T* pics, const T* ref, const T* shx, const T* shy, size2_t one_size, size_t ref_slices, size_t batch_size)
{
	size_t block_size = 1024;
	size_t grid_size(div_up(one_size.area() * ref_slices * batch_size, block_size));
	hadamard <<<grid_size, block_size >>> (pics, ref, shx, shy, one_size, ref_slices, batch_size);
}

template void run_hadamard<cufftComplex>(cufftComplex* A, const cufftComplex* B, const cufftComplex* shx, const cufftComplex* shy, size2_t one_size, size_t ref_slices, size_t batch_size);
template void run_hadamard<cufftDoubleComplex>(cufftDoubleComplex* A, const cufftDoubleComplex* B, const cufftDoubleComplex* shx, const cufftDoubleComplex* shy, size2_t one_size, size_t ref_slices, size_t batch_size);

template<typename T>
void run_hadamard(T* pics, const T* ref, size2_t one_size, size_t ref_slices, size_t batch_size)
{
	size_t block_size = 1024;
	size_t grid_size(div_up(one_size.area() * ref_slices * batch_size, block_size));
	hadamard << <grid_size, block_size >> > (pics, ref, one_size, ref_slices, batch_size);
}

template void run_hadamard<cufftComplex>(cufftComplex* A, const cufftComplex* B, size2_t one_size, size_t ref_slices, size_t batch_size);
template void run_hadamard<cufftDoubleComplex>(cufftDoubleComplex* A, const cufftDoubleComplex* B, size2_t one_size, size_t ref_slices, size_t batch_size);


template<typename T>
__global__ void finalize_fft(const T* in, T* __restrict__ out, size2_t out_size, size_t batch_size)
{
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	size2_t in_size{ out_size.x + 3, out_size.y + 1 };
	
	size_t slice_tid = i % out_size.area();
	size_t slice_num = i / out_size.area();
	size2_t slice_pos = { slice_tid % out_size.x, slice_tid / out_size.x };

	if (slice_num >= batch_size)
		return;

	size2_t in_pos = (slice_pos + ((out_size + 1) / 2 + 1)) % (out_size+1);

	out[i] = in[slice_num * in_size.area() + in_pos.pos(in_size.x)];
}

template<typename T>
void run_finalize_fft(const T* in, T* out, size2_t out_size, size_t batch_size)
{
	size_t block_size = 1024;
	size_t grid_size(div_up(out_size.area() * batch_size, block_size));
	finalize_fft <<<grid_size, block_size >>> (in, out, out_size, batch_size);
}

template void run_finalize_fft<double>(const double* in, double* out, size2_t out_size, size_t batch_size);
template void run_finalize_fft<float>(const float* in, float* out, size2_t out_size, size_t batch_size);

}
