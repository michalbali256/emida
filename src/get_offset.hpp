#pragma once


#include "subpixel_max.hpp"
#include "kernels.cuh"

namespace emida
{

template<typename T>
void subtract_mean(T * pic, size_t size)
{
	T sum = 0;
	for (size_t i = 0; i < size; ++i)
		sum += pic[i];

	T avg = sum / size;

	for (size_t i = 0; i < size; ++i)
		pic[i] -= avg;
}

template<typename T>
inline vec2<T> get_offset(T* pic, T* temp, size_t cols, size_t rows)
{
	size_t size = cols * rows;

	subtract_mean(pic, size);
	subtract_mean(temp, size);

	T* cu_pic = vector_to_device(pic, size);
	T* cu_temp = vector_to_device(temp, size);

	auto hann_x = hanning<T>(cols);
	auto hann_y = hanning<T>(rows);

	T* cu_hann_x = vector_to_device(hann_x);
	T* cu_hann_y = vector_to_device(hann_y);

	run_hanning(cu_pic, cu_hann_x, cu_hann_y, cols, rows);
	run_hanning(cu_temp, cu_hann_x, cu_hann_y, cols, rows);

	CUCH(cudaGetLastError());
	CUCH(cudaDeviceSynchronize());

	T* cu_cross_res;
	size_t cross_cols = 2 * cols - 1;
	size_t cross_rows = 2 * rows - 1;
	size_t cross_size = cross_cols * cross_rows;

	CUCH(cudaMalloc(&cu_cross_res, cross_size * sizeof(T)));

	run_cross_corr(cu_pic, cu_temp, cu_cross_res, cols, rows);

	CUCH(cudaGetLastError());
	CUCH(cudaDeviceSynchronize());

	auto cross_data = device_to_vector(cu_cross_res, cross_size);

	size_t maxarg_block_size = 1024;
	data_index<T> * cu_maxes;
	size_t maxes_size = div_up(cross_size, maxarg_block_size);
	CUCH(cudaMalloc(&cu_maxes, maxes_size * sizeof(data_index<T>)));
	run_maxarg_reduce(cu_cross_res, cu_maxes, cross_size, maxarg_block_size);

	CUCH(cudaGetLastError());
	CUCH(cudaDeviceSynchronize());

	
	std::vector<data_index<T>> maxes = device_to_vector(cu_maxes, maxes_size);
	
	size_t max_res_i = 0;
	for (size_t i = 1; i < maxes.size(); ++i)
	{
		if (maxes[i].data > maxes[max_res_i].data)
			max_res_i = i;
	}
	size_t max_i = maxes[max_res_i].index;


	size_t max_x = max_i % cross_cols;
	size_t max_y = max_i / cross_cols;

	constexpr int s = 3;
	constexpr int r = (s-1)/2;

	size_t neigh_size = s * s;
	T* cu_neighbors;
	CUCH(cudaMalloc(&cu_neighbors, neigh_size * sizeof(T)));
	run_extract_neighbors<T, s>(cu_cross_res, cu_neighbors, max_x, max_y, cross_cols, cross_rows);

	CUCH(cudaGetLastError());
	CUCH(cudaDeviceSynchronize());

	std::vector<T> neighbors = device_to_vector(cu_neighbors, neigh_size);

	auto subp_offset = subpixel_max_serial<T, s>(neighbors.data());

	vec2<T> res;
	res.x = (int)max_x - (int)cols + 1 - r + subp_offset.x;
	res.y = (int)max_y - (int)rows + 1 - r + subp_offset.y;

	return res;
}

}