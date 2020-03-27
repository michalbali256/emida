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
	vec2<size_t> size_;
	vec2<size_t> res_size_;
	size_t  batch_size_;
	
	std::vector<T> res_;
public:
	
	void prepare(const T * a, const T * b, vec2<size_t> size, vec2<size_t> res_size, size_t batch_size)
	{
		size_ = size;
		res_size_ = res_size;
		batch_size_ = batch_size;
		
		size_t pic_size = size_.area() * batch_size;


		CUCH(cudaMalloc(&cu_a_, pic_size * sizeof(T)));
		CUCH(cudaMalloc(&cu_b_, pic_size * sizeof(T)));
		CUCH(cudaMalloc(&cu_res_, res_size_.area() * batch_size * sizeof(T)));

		CUCH(cudaMemcpy(cu_a_, a, pic_size * sizeof(T), cudaMemcpyHostToDevice));
		CUCH(cudaMemcpy(cu_b_, b, pic_size * sizeof(T), cudaMemcpyHostToDevice));

		res_.resize(res_size_.area() * batch_size);
		
	}

	void run()
	{
		run_cross_corr<T, T>(cu_a_, cu_b_, cu_res_, size_, res_size_, batch_size_);

		
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
