#pragma once

#include <cstdint>
#include <cmath>
#include <cuda_runtime.h>

#include "matrix.hpp"
#include "kernels.cuh"


namespace emida
{

template<typename T>
class algorithm_hanning
{
	T * cu_pic_, *cu_hann_x_, *cu_hann_y_;

	size_t cols_, rows_, batch_size_, size_;

	std::vector<T> res_;
public:
	
	void prepare(const T * pic, size_t cols, size_t rows, size_t b_size)
	{
		batch_size_ = b_size;
		cols_ = cols;
		rows_ = rows;
		size_ = cols * rows * b_size;
		cu_pic_ = vector_to_device(pic, size_);

		auto hann_window_x = generate_hanning<T>(cols);
		auto hann_window_y = generate_hanning<T>(rows);
		cu_hann_x_ = vector_to_device(hann_window_x);
		cu_hann_y_ = vector_to_device(hann_window_y);
	}

	void run()
	{
		run_hanning<double>(cu_pic_, cu_hann_x_, cu_hann_y_, cols_, rows_, batch_size_);
		
		CUCH(cudaDeviceSynchronize());
		CUCH(cudaGetLastError());
	}

	void finalize()
	{
		res_.resize(size_);
		CUCH(cudaMemcpy(res_.data(), cu_pic_, res_.size() * sizeof(T), cudaMemcpyDeviceToHost));
	}

	const std::vector<T> & result()
	{
		return res_;
	}
};

}
