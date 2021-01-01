
#include "kernels.cuh"
#include "host_helpers.hpp"
#include "stopwatch.hpp"
#include "cufft.h"
#include "cufft_helpers.hpp"

using namespace emida;

int main(int argc, char** argv)
{
	size_t size = std::stol(argv[1]);
	size_t count = std::stol(argv[2]);
	size_t batch = std::stol(argv[3]);
	size2_t half_size{ size * 2 + 1, size * 4 };
	cufftDoubleComplex* pics = emida::cuda_malloc<cufftDoubleComplex>(half_size.area() * count * batch);
	cufftDoubleComplex* ref = emida::cuda_malloc<cufftDoubleComplex>(half_size.area() * count * batch);

	cufftDoubleComplex* pics2 = emida::cuda_malloc<cufftDoubleComplex>(half_size.area() * count * batch);
	cufftDoubleComplex* ref2 = emida::cuda_malloc<cufftDoubleComplex>(half_size.area() * count * batch);


	int fft_size_[2] = { (int)size * 2, (int)size * 2 };
	cufftHandle plan_;

	FFTCH(cufftPlanMany(&plan_, 2, fft_size_,
		NULL, 1, 0,
		NULL, 1, 0,
		fft_type_R2C<double>(), (int)(count * batch)));

	stopwatch::global_activate = true;
	stopwatch sw;
	sw.zero();
	for (size_t i = 0; i < 10; ++i)
	{
		fft_real_to_complex(plan_, (double*) pics2, pics);
		CUCH(cudaDeviceSynchronize()); sw.tick("R2C: ");

		run_hadamard(pics, ref, half_size, count, batch);
		CUCH(cudaGetLastError());
		CUCH(cudaDeviceSynchronize());
		sw.tick("Multiply");
	}

	/*
	sw.zero();
	for (size_t i = 0; i < 10; ++i)
	{
		run_hadamard(pics, ref, half_size, count, batch);
		CUCH(cudaGetLastError());
		CUCH(cudaDeviceSynchronize()); sw.tick("Multiply2");

		run_hadamard(pics2, ref2, half_size, count, batch);
		CUCH(cudaGetLastError());
		CUCH(cudaDeviceSynchronize());
		sw.tick("Multiply3");
	}
	*/
	sw.write_durations();
	
}
