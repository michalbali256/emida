#pragma once

#include <cstdint>
#include <cmath>
#include <cuda_runtime.h>

#include "matrix.hpp"
#include "cross_corr.cuh"


#define CUCH(status) emida::cuda_check(status, __LINE__, __FILE__, #status)

namespace emida
{

inline void cuda_check(cudaError_t status, int line, const char* src_filename, const char* line_str = nullptr)
{
	if (status != cudaSuccess)
	{
		std::stringstream ss;
		ss << "CUDA Error " << status << ":" << cudaGetErrorString(status) << " in " << src_filename << " (" << line << "):" << line_str << "\n";
		std::cout << ss.str();
		throw std::exception(ss.str().c_str());
	}
}



class algorithm_cross_corr
{

	
	int* cu_a, * cu_b, * cu_res;
	int n;
	size_t res_n, res_size;
	matrix<int> res;
public:
	
	void prepare(const matrix<int>& a, const matrix<int>& b)
	{
		n = a.n;
		res_n = 2 * n - 1;
		res_size = res_n * res_n;
		CUCH(cudaMalloc(&cu_a, a.data.size() * sizeof(int)));
		CUCH(cudaMalloc(&cu_b, b.data.size() * sizeof(int)));
		CUCH(cudaMalloc(&cu_res, res_size * sizeof(int)));

		CUCH(cudaMemcpy(cu_a, a.data.data(), a.data.size() * sizeof(int), cudaMemcpyHostToDevice));
		CUCH(cudaMemcpy(cu_b, b.data.data(), b.data.size() * sizeof(int), cudaMemcpyHostToDevice));

		res.data.resize(res_size);
		
	}

	void run()
	{
		run_cross_corr<int, int>(cu_a, cu_b, cu_res, n, n);

		
		CUCH(cudaDeviceSynchronize());
		CUCH(cudaGetLastError());
	}

	void finalize()
	{
		CUCH(cudaMemcpy(res.data.data(), cu_res, res.data.size() * sizeof(int), cudaMemcpyDeviceToHost));
	}

	const matrix<int> & result()
	{
		return res;
	}
};

}
