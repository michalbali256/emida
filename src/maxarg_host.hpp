#pragma once

#include <cstdint>
#include <cmath>
#include <cuda_runtime.h>

#include "matrix.hpp"
#include "kernels.cuh"


namespace emida
{

template<typename T>
class algorithm_maxarg
{
	size_t size_, batch_size_, grid_size_, one_pic_blocks_;
	T* cu_data_;
	data_index<T>* cu_res_;
	std::vector<data_index<T>> res_;
	constexpr static size_t block_size_ = 1024;
	std::vector<size_t> max_i_;
public:
	
	void prepare(const std::vector<T>& data, size_t size, size_t batch_size)
	{
		size_ = size;
		batch_size_ = batch_size;
		cu_data_ = vector_to_device(data);

		one_pic_blocks_ = div_up(size, block_size_);
		grid_size_ = one_pic_blocks_ * batch_size;
		res_.resize(grid_size_);
		cu_res_ = vector_to_device(res_);

		
	}

	void run()
	{
		run_maxarg_reduce<T>(cu_data_, cu_res_, size_, block_size_, batch_size_);
		
		CUCH(cudaDeviceSynchronize());
		CUCH(cudaGetLastError());

	}

	void finalize()
	{
		max_i_.resize(batch_size_, 0);
		CUCH(cudaMemcpy(res_.data(), cu_res_, res_.size() * sizeof(data_index<T>), cudaMemcpyDeviceToHost));
		for (size_t b = 0; b < batch_size_; ++b)
		{
			size_t max_res_i = b * one_pic_blocks_;
			for (size_t i = max_res_i + 1; i < (b + 1) * one_pic_blocks_; ++i)
			{
				if (res_[i].data > res_[max_res_i].data)
					max_res_i = i;
			}
			max_i_[b] = res_[max_res_i].index;
		}
	}

	std::vector<size_t> result()
	{
		return max_i_;
	}
};

}
