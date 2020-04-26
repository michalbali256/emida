#pragma once

#include <cstdint>

#include "common.hpp"

#ifdef __INTELLISENSE__
void __syncthreads() {};
#include "device_launch_parameters.h"

#endif

namespace emida
{

template<typename T, typename RES>
void run_cross_corr(const T* pic_a, const T* pic_b, RES* res, vec2<size_t> size, vec2<size_t> res_size, size_t batch_size);

template<typename T>
void run_prepare_pics(T* pic, const T* hanning_x, const T* hanning_y, const T* sums, size2_t size, size_t batch_size);

template<typename IN, typename OUT>
void run_prepare_pics(
	const IN* pic,
	OUT* slices,
	const OUT* hanning_x,
	const OUT* hanning_y,
	const OUT* sums,
	const size2_t* begins,
	size2_t src_size,
	size2_t slice_size,
	size_t batch_size);

template<typename T>
void run_maxarg_reduce(const T* data, data_index<T>* maxes, size_t size, size_t block_size, size_t batch_size);

template<typename T, int s>
void run_extract_neighbors(const T* data, const vec2<size_t>* max_i, T* neighbors, size_t cols, size_t rows, size_t batch_size);

//computes sums of slices of the same size packed in one array
template<typename T>
void run_sum(const T* data, T* sums, size_t size, size_t batch_size);

//computes sums of slices of the same size with specified positions (begins) in a big picture (data)
template<typename T, typename RES>
void run_sum(const T* data, RES* sums, const size2_t* begins, size2_t src_size, size2_t slice_size, size_t batch_size);

}