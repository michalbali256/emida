#include "cuda.h"
#include "cuda_runtime.h"
#include "cufft.h"

#include "kernels.cuh"

namespace emida
{

template<typename T>
__global__ void hadamard(T * __restrict__ A, const T * __restrict__ B, size_t size)
{
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= size)
		return;

	A[i] = { A[i].x * B[i].x + A[i].y * B[i].y,
		- A[i].x * B[i].y + A[i].y * B[i].x };
}

template<typename T>
void run_hadamard(T* A, const T* B, size_t size)
{
	size_t block_size = 1024;
	size_t grid_size(div_up(size, block_size));
	hadamard <<<grid_size, block_size >>> (A, B, size);
}


template void run_hadamard<cufftComplex>(cufftComplex* A, const cufftComplex* B, size_t size);
template void run_hadamard<cufftDoubleComplex>(cufftDoubleComplex* A, const cufftDoubleComplex* B, size_t size);

}