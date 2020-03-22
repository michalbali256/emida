#pragma once


#include "subpixel_max.hpp"
#include "kernels.cuh"
#include "subtract_mean.hpp"
#include "maxarg.hpp"

namespace emida
{

//gets two pictures with size cols x rows and returns subpixel offset between them
template<typename T>
inline std::vector<vec2<T>> get_offset(T* pic, T* temp, size_t cols, size_t rows, size_t b_size)
{
	size_t one_size = cols * rows;

	subtract_mean(pic, one_size, b_size);
	subtract_mean(temp, one_size, b_size);
	
	T* cu_pic = vector_to_device(pic, one_size);
	T* cu_temp = vector_to_device(temp, one_size);

	auto hann_x = generate_hanning<T>(cols);
	auto hann_y = generate_hanning<T>(rows);

	T* cu_hann_x = vector_to_device(hann_x);
	T* cu_hann_y = vector_to_device(hann_y);

	run_hanning(cu_pic, cu_hann_x, cu_hann_y, cols, rows, b_size);
	run_hanning(cu_temp, cu_hann_x, cu_hann_y, cols, rows, b_size);

	CUCH(cudaGetLastError());
	CUCH(cudaDeviceSynchronize());

	T* cu_cross_res;
	size_t cross_cols = 2 * cols - 1;
	size_t cross_rows = 2 * rows - 1;
	size_t cross_size = cross_cols * cross_rows;

	CUCH(cudaMalloc(&cu_cross_res, cross_size * sizeof(T)));

	run_cross_corr(cu_temp, cu_pic, cu_cross_res, cols, rows, b_size);

	CUCH(cudaGetLastError());
	CUCH(cudaDeviceSynchronize());

	std::vector<vec2<size_t>> maxes_i = get_maxarg(cu_cross_res, cross_cols, cross_rows, b_size);

	constexpr int s = 3;
	constexpr int r = (s-1)/2;

	size_t neigh_size = s * s * b_size;
	T* cu_neighbors;
	CUCH(cudaMalloc(&cu_neighbors, neigh_size * sizeof(T)));

	auto cu_maxes_i = vector_to_device(maxes_i);

	run_extract_neighbors<T, s>(cu_cross_res, cu_maxes_i, cu_neighbors, cross_cols, cross_rows, b_size);

	CUCH(cudaGetLastError());
	CUCH(cudaDeviceSynchronize());

	std::vector<T> neighbors = device_to_vector(cu_neighbors, neigh_size);

	auto subp_offset = subpixel_max_serial<T, s>(neighbors.data(), b_size);
	
	std::vector<vec2<T>> res(b_size);
	for (size_t i = 0; i < b_size; ++i)
	{
		res[i].x = (int)maxes_i[i].x - (int)cols + 1 - r + subp_offset[i].x;
		res[i].y = (int)maxes_i[i].y - (int)rows + 1 - r + subp_offset[i].y;
	}
	return res;
}

}