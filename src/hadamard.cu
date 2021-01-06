#include "cuda.h"
#include "cuda_runtime.h"
#include "cufft.h"

#include "kernels.cuh"

namespace emida
{

//This file contains the implementation of complex matrix
//element-by-element multiplication (also called hadamard product)

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

template<typename T>
__global__ void hadamard(T* __restrict__ pics,
	const T* __restrict__ ref,
	size2_t one_size,
	esize_t ref_slices,
	esize_t batch_size)
{
	esize_t ref_i = blockIdx.x * blockDim.x + threadIdx.x;

	if (ref_i >= one_size.area() * ref_slices)
		return;

	for(esize_t i = ref_i; i < one_size.area() * ref_slices * batch_size; i += one_size.area() * ref_slices)
		pics[i] = { (pics[i].x * ref[ref_i].x + pics[i].y * ref[ref_i].y),
			(-pics[i].x * ref[ref_i].y + pics[i].y * ref[ref_i].x) };
}

template<typename T>
void run_hadamard(T* pics, const T* ref, size2_t one_size, esize_t ref_slices, esize_t batch_size)
{
	esize_t block_size = 1024;
	esize_t grid_size(div_up(one_size.area() * ref_slices, block_size));
	hadamard <<<grid_size, block_size >>> (pics, ref, one_size, ref_slices, batch_size);
}

template void run_hadamard<cufftComplex>(cufftComplex* A, const cufftComplex* B, size2_t one_size, esize_t ref_slices, esize_t batch_size);
template void run_hadamard<cufftDoubleComplex>(cufftDoubleComplex* A, const cufftDoubleComplex* B, size2_t one_size, esize_t ref_slices, esize_t batch_size);

}
