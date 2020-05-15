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
__global__ void hadamard(T * __restrict__ A, const T * __restrict__ B, const T* __restrict__ shx, const T* __restrict__ shy, size2_t one_size, size_t batch_size)
{
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	
	size_t slice_tid = i % one_size.area();
	size_t slice_num = i / one_size.area();
	size2_t slice_pos = { slice_tid % one_size.x, slice_tid / one_size.x };

	if (slice_num >= batch_size)
		return;
	
	T mul = { (A[i].x * B[i].x + A[i].y * B[i].y),
		(-A[i].x * B[i].y + A[i].y * B[i].x)};
	//mul = complex_mul(mul, shx[slice_pos.x]);
	//mul = complex_mul(mul, shy[slice_pos.y]);
	//size_t num_elements = (one_size.x - 1) * 2 * one_size.y;
	//mul = { mul.x / num_elements, mul.y / num_elements };
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
void run_hadamard(T* A, const T* B, const T* shx, const T* shy, size2_t one_size, size_t batch_size)
{
	size_t block_size = 1024;
	size_t grid_size(div_up(one_size.area() * batch_size, block_size));
	hadamard <<<grid_size, block_size >>> (A, B, shx, shy, one_size, batch_size);
}


template void run_hadamard<cufftComplex>(cufftComplex* A, const cufftComplex* B, const cufftComplex* shx, const cufftComplex* shy, size2_t one_size, size_t batch_size);
template void run_hadamard<cufftDoubleComplex>(cufftDoubleComplex* A, const cufftDoubleComplex* B, const cufftDoubleComplex* shx, const cufftDoubleComplex* shy, size2_t one_size, size_t batch_size);

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
