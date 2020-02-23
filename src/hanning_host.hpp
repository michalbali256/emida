#pragma once

#include <cstdint>
#include <cmath>
#include <cuda_runtime.h>

#include "matrix.hpp"
#include "kernels.cuh"


namespace emida
{

class algorithm_hanning
{
	double* cu_pic, *cu_hann_x, *cu_hann_y;

	int n;
	matrix<double> res;
public:
	
	void prepare(const matrix<double>& pic)
	{
		n = pic.n;

		cu_pic = vector_to_device(pic.data);

		auto hann_window = hanning<double>(n);
		cu_hann_x = cu_hann_y = vector_to_device(hann_window);
	}

	void run()
	{
		run_hanning<double>(cu_pic, cu_hann_x, cu_hann_y, n, n);
		
		CUCH(cudaDeviceSynchronize());
		CUCH(cudaGetLastError());
	}

	void finalize()
	{
		res.n = n;
		res.data.resize(n * n);
		CUCH(cudaMemcpy(res.data.data(), cu_pic, res.data.size() * sizeof(double), cudaMemcpyDeviceToHost));
	}

	const matrix<double> & result()
	{
		return res;
	}
};

}
