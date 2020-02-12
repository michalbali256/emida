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
	size_t size;
	T* cu_data;
	data_index<T>* cu_res;
	std::vector<data_index<T>> res;
	const size_t block_size = 1024;
	size_t max_i = 0;
public:
	
	void prepare(const std::vector<T>& data)
	{
		size = data.size();
		cu_data = vector_to_host(data);

		res.resize(div_up(data.size(), block_size));
		cu_res = vector_to_host(res);
	}

	void run()
	{
		run_maxarg_reduce<T>(cu_data, cu_res, size, block_size);
		
		CUCH(cudaDeviceSynchronize());
		CUCH(cudaGetLastError());

	}

	void finalize()
	{
		CUCH(cudaMemcpy(res.data(), cu_res, res.size() * sizeof(data_index<T>), cudaMemcpyDeviceToHost));
		size_t max_res_i = 0;
		for (size_t i = 1; i < res.size(); ++i)
		{
			if (res[i].data > res[max_res_i].data)
				max_res_i = i;
		}
		max_i = res[max_res_i].index;
	}

	size_t result()
	{
		return max_i;
	}
};

}
