#pragma once

#include <cstdint>

#include "common.hpp"

#ifdef __INTELLISENSE__
void __syncthreads() {};
#include "device_launch_parameters.h"

#endif

namespace emida
{

struct cross_res_pos_policy_id
{
	static __device__ __inline__ esize_t pos(esize_t i, esize_t pic_num, size2_t slice_pos, size2_t slice_size)
	{
		return i;
	}
};

struct cross_res_pos_policy_fft
{
	static __device__ __inline__ esize_t pos(esize_t i, esize_t pic_num, size2_t slice_pos, size2_t slice_size)
	{
		size2_t in_size{ slice_size.x + 3, slice_size.y + 1 };
		size2_t in_pos = (slice_pos + ((slice_size + 1) / 2 + 1)) % (slice_size + 1);
		return pic_num * in_size.area() + in_pos.pos(in_size.x);
	}
};


template<typename T, typename RES>
void run_cross_corr(const T* pics,
	const T* ref,
	RES* res,
	size2_t size,
	size2_t res_size,
	esize_t ref_slices,
	esize_t batch_size);

template<typename T, typename RES>
void run_cross_corr_r(const T* pics,
	const T* ref,
	RES* res,
	size2_t size,
	size2_t res_size,
	esize_t ref_slices,
	esize_t batch_size);

template<typename T, typename RES>
void run_cross_corr_opt(const T* pic_a,
	const T* pic_b,
	RES* res,
	size2_t size,
	size2_t res_size,
	esize_t ref_slices,
	esize_t batch_size);

template<typename T, typename RES>
void run_cross_corr_opt_tr(const T* pic_a,
	const T* pic_b,
	RES* res,
	size2_t size,
	size2_t res_size,
	esize_t ref_slices,
	esize_t batch_size);

template<typename T>
void run_prepare_pics(T* pic, const T* hanning_x, const T* hanning_y, const T* sums, size2_t size, esize_t batch_size);

template<typename IN, typename OUT, typename S = OUT>
void run_prepare_pics(
	const IN* pic,
	OUT* slices,
	const OUT* hanning_x,
	const OUT* hanning_y,
	const S* sums,
	const size2_t* begins,
	size2_t src_size,
	size2_t slice_size,
	size2_t out_size,
	esize_t begins_size,
	esize_t batch_size);

template<typename T, class pos_policy = cross_res_pos_policy_id>
void run_maxarg_reduce(const T* data, data_index<T>* maxes, size2_t* maxarg, size2_t size, esize_t block_size, esize_t batch_size);

template<typename T, class pos_policy = cross_res_pos_policy_id>
void run_extract_neighbors(const T* data, const size2_t* max_i, T* neighbors, int s, size2_t src_size, esize_t batch_size);

//computes sums of slices of the same size packed in one array
template<typename T>
void run_sum(const T* data, T* sums, esize_t size, esize_t batch_size);

//computes sums of slices of the same size with specified positions (begins) in a big picture (data)
template<typename T, typename RES>
void run_sum(const T* data, RES* sums, const size2_t* begins, size2_t src_size, size2_t slice_size, esize_t begins_size, esize_t batch_size);

template<typename T>
void run_hadamard(T* A, const T* B, const T* shx, const T* shy, size2_t one_size, esize_t ref_slices, esize_t batch_size);

template<typename T>
void run_hadamard(T* pics, const T* ref, size2_t one_size, esize_t ref_slices, esize_t batch_size);

template<typename T>
void run_finalize_fft(const T* in, T* out, size2_t out_size, esize_t batch_size);

}
