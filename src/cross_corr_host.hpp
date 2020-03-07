#pragma once

#include <cstdint>
#include <cmath>
#include <cuda_runtime.h>

#include "matrix.hpp"
#include "kernels.cuh"


namespace emida
{

template<typename T>
class algorithm_cross_corr
{
	T* cu_a_, * cu_b_, * cu_res_;
	size_t cols_, rows_, batch_size_;
	size_t res_cols_, res_rows_, res_size_;
	std::vector<T> res_;
public:
	
	void prepare(const T * a, const T * b, size_t cols, size_t rows, size_t batch_size)
	{
		cols_ = cols;
		rows_ = rows;
		batch_size_ = batch_size;
		
		size_t size = cols * rows * batch_size;

		res_cols_ = 2 * cols - 1;
		res_rows_ = 2 * rows - 1;
		res_size_ = res_cols_ * res_rows_;

		CUCH(cudaMalloc(&cu_a_, size * sizeof(T)));
		CUCH(cudaMalloc(&cu_b_, size * sizeof(T)));
		CUCH(cudaMalloc(&cu_res_, res_size_ * batch_size * sizeof(T)));

		CUCH(cudaMemcpy(cu_a_, a, size * sizeof(T), cudaMemcpyHostToDevice));
		CUCH(cudaMemcpy(cu_b_, b, size * sizeof(T), cudaMemcpyHostToDevice));

		res_.resize(res_size_ * batch_size);
		
	}

	void run()
	{
		run_cross_corr<T, T>(cu_a_, cu_b_, cu_res_, cols_, rows_, batch_size_);

		
		CUCH(cudaDeviceSynchronize());
		CUCH(cudaGetLastError());
	}

	void finalize()
	{
		CUCH(cudaMemcpy(res_.data(), cu_res_, res_.size() * sizeof(T), cudaMemcpyDeviceToHost));
	}

	const std::vector<T> & result()
	{
		return res_;
	}
};

}
