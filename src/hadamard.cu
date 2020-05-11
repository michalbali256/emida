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
__global__ void hadamard(T * __restrict__ A, const T * __restrict__ B, size2_t one_size, size_t batch_size)
{
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	
	size_t slice_tid = i % one_size.area();
	size_t slice_num = i / one_size.area();
	size2_t slice_pos = { slice_tid % one_size.x, slice_tid / one_size.x };

	if (slice_num >= batch_size)
		return;
	
	T mul = { (A[i].x * B[i].x + A[i].y * B[i].y),
		(-A[i].x * B[i].y + A[i].y * B[i].x)};
	size_t xn = (one_size.x - 1) * 2;
	size_t my = one_size.y / 2 + 1;
	T shx = { cos(2 * PI / xn * slice_pos.x * one_size.x), sin(2 * PI / xn * slice_pos.x * one_size.x) };
	T shy = { cos(2 * PI / one_size.y * slice_pos.y * my), sin(2 * PI / one_size.y * slice_pos.y * my) };
	mul = complex_mul(mul, shx);
	mul = complex_mul(mul, shy);
	mul = { mul.x / (xn*one_size.y), mul.y / (xn * one_size.y) };
	A[i] = mul;
}

/*template<typename T>
__global__ void hadamard(T* __restrict__ A, const T* __restrict__ B, size2_t one_size, size_t batch_size)
{
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= one_size.area() * batch_size)
		return;

	
	A[i] = { (A[i].x * B[i].x + A[i].y * B[i].y),
		(A[i].y * B[i].x - A[i].x * B[i].y) };
}*/

template<typename T>
void run_hadamard(T* A, const T* B, size2_t one_size, size_t batch_size)
{
	size_t block_size = 1024;
	size_t grid_size(div_up(one_size.area() * batch_size, block_size));
	hadamard <<<grid_size, block_size >>> (A, B, one_size, batch_size);
}


template void run_hadamard<cufftComplex>(cufftComplex* A, const cufftComplex* B, size2_t one_size, size_t batch_size);
template void run_hadamard<cufftDoubleComplex>(cufftDoubleComplex* A, const cufftDoubleComplex* B, size2_t one_size, size_t batch_size);

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

	out[i] = in[slice_num * in_size.area() + slice_pos.pos(in_size.x)];
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