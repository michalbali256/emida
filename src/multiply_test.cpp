
#include "kernels.cuh"
#include "host_helpers.hpp"
#include "stopwatch.hpp"
#include "cufft.h"

using namespace emida;

int main(int argc, char** argv)
{
	size_t size = std::stol(argv[1]);
	size_t count = std::stol(argv[2]);
	size_t batch = std::stol(argv[3]);
	size2_t half_size{ size * 2 + 1, size * 4 };
	cufftComplex* pics = emida::cuda_malloc<cufftComplex>(half_size.area() * count * batch);
	cufftComplex* ref = emida::cuda_malloc<cufftComplex>(half_size.area() * count * batch);

	stopwatch::global_activate = true;
	stopwatch sw;
	sw.zero();
	for (size_t i = 0; i < 50; ++i)
	{
		run_hadamard(pics, ref, half_size, count, batch);
		CUCH(cudaGetLastError());
		CUCH(cudaDeviceSynchronize());
		sw.tick("Multiply");
	}

	sw.write_durations();
	
}
