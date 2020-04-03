#pragma once


#include "subpixel_max.hpp"
#include "kernels.cuh"
#include "subtract_mean.hpp"
#include "maxarg.hpp"
#include "stopwatch.hpp"

namespace emida
{

template<typename T>
inline std::vector<vec2<T>> get_offset(T* pic, T* temp, vec2<size_t> size, size_t b_size)
{
	return get_offset(pic, temp, size, size * 2 - 1, b_size);
}

//gets two pictures with size cols x rows and returns subpixel offset between them
template<typename T>
inline std::vector<vec2<T>> get_offset(T* pic, T* temp, vec2<size_t> size, vec2<size_t> cross_size, size_t b_size)
{
	START_STOPWATCH();

	size_t one_size = size.area();
	subtract_mean(pic, one_size, b_size);
	subtract_mean(temp, one_size, b_size); TICK("Subtract: ");

	T* cu_pic = vector_to_device(pic, one_size * b_size);
	T* cu_temp = vector_to_device(temp, one_size * b_size); TICK("Temp and pic to device: ");

	auto hann_x = generate_hanning<T>(size.x);
	auto hann_y = generate_hanning<T>(size.y); TICK("Generate hanning: ");

	T* cu_hann_x = vector_to_device(hann_x);
	T* cu_hann_y = vector_to_device(hann_y); TICK("Hanning to device: ");

	run_hanning(cu_pic, cu_hann_x, cu_hann_y, size.x, size.y, b_size);
	run_hanning(cu_temp, cu_hann_x, cu_hann_y, size.x, size.y, b_size);

	CUCH(cudaGetLastError());
	CUCH(cudaDeviceSynchronize()); TICK("Run hanning: ");

	T* cu_cross_res;

	CUCH(cudaMalloc(&cu_cross_res, cross_size.area() * b_size * sizeof(T))); TICK("Cross result malloc: ");

	run_cross_corr(cu_temp, cu_pic, cu_cross_res, size, cross_size, b_size);

	CUCH(cudaGetLastError());
	CUCH(cudaDeviceSynchronize()); TICK("Run cross corr: ");

	constexpr int s = 3;
	constexpr int r = (s - 1) / 2;

	size_t neigh_size = s * s * b_size;
	T* cu_neighbors;
	std::vector<vec2<size_t>> maxes_i;
	if (cross_size.x == s && cross_size.y == s)
	{
		cu_neighbors = cu_cross_res;
	}
	else
	{
		maxes_i = get_maxarg(cu_cross_res, cross_size.x, cross_size.y, b_size); TICK("Get maxarg: ");
		
		CUCH(cudaMalloc(&cu_neighbors, neigh_size * sizeof(T))); TICK("Neighbors malloc: ");

		auto cu_maxes_i = vector_to_device(maxes_i); TICK("Maxes transfer: ");

		run_extract_neighbors<T, s>(cu_cross_res, cu_maxes_i, cu_neighbors, cross_size.x, cross_size.y, b_size);

		CUCH(cudaGetLastError());
		CUCH(cudaDeviceSynchronize()); TICK("Run extract neigh: ");
	}

	std::vector<T> neighbors = device_to_vector(cu_neighbors, neigh_size); TICK("Transfer neighbors: ");

	auto subp_offset = subpixel_max_serial<T, s>(neighbors.data(), b_size); TICK("Subpixel max: ");
	
	std::vector<vec2<T>> res(b_size);
	if (cross_size.x == s && cross_size.y == s)
	{
		for (size_t i = 0; i < b_size; ++i)
		{
			res[i] = subp_offset[i] - r;
		}
	}
	else
	{
		for (size_t i = 0; i < b_size; ++i)
		{

			res[i].x = (int)maxes_i[i].x - ((int)cross_size.x / 2) - r + subp_offset[i].x;
			res[i].y = (int)maxes_i[i].y - ((int)cross_size.y / 2) - r + subp_offset[i].y;
		}
	}
	TOTAL();
	return res;
}

}