#pragma once

#include <cstdint>

#include "common.hpp"

#ifdef __INTELLISENSE__
void __syncthreads() {};

#endif

namespace emida
{

template<typename T, typename RES>
void run_cross_corr(const T* pic_a, const T* pic_b, RES* res, size_t cols, size_t rows, size_t batch_size);

template<typename T>
void run_hanning(T* pic, const T* hanning_x, const T* hanning_y, size_t cols, size_t rows, size_t batch_size);

template<typename T>
void run_maxarg_reduce(const T* data, data_index<T>* maxes, size_t size, size_t block_size, size_t batch_size);

template<typename T, int s>
void run_extract_neighbors(const T* data, const vec2<size_t>* max_i, T* neighbors, size_t cols, size_t rows, size_t batch_size);

}