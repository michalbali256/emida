#pragma once

namespace emida
{

template<typename T>
inline std::vector<vec2<size_t>> get_maxarg(const T* cu_data, size_t cols, size_t rows, size_t batch_size)
{
	size_t maxarg_block_size = 1024;
	data_index<T>* cu_maxes;
	size_t size = cols * rows;
	size_t one_pic_blocks = div_up(size, maxarg_block_size);
	size_t maxes_size = one_pic_blocks * batch_size;

	CUCH(cudaMalloc(&cu_maxes, maxes_size * sizeof(data_index<T>)));
	run_maxarg_reduce(cu_data, cu_maxes, size, maxarg_block_size, batch_size);

	CUCH(cudaGetLastError());
	CUCH(cudaDeviceSynchronize());

	std::vector<data_index<T>> maxes = device_to_vector(cu_maxes, maxes_size);

	std::vector<vec2<size_t>> res(batch_size);

	for (size_t b = 0; b < batch_size; ++b)
	{
		size_t max_res_i = b * one_pic_blocks;
		for (size_t i = max_res_i + 1; i < (b + 1) * one_pic_blocks; ++i)
		{
			if (maxes[i].data > maxes[max_res_i].data)
				max_res_i = i;
		}
		size_t max_i = maxes[max_res_i].index - b * size;
		res[b].x = max_i % cols;
		res[b].y = max_i / cols;
	}

	return res;
}


}