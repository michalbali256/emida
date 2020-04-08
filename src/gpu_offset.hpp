#pragma once


#include "subpixel_max.hpp"
#include "kernels.cuh"
#include "subtract_mean.hpp"
#include "stopwatch.hpp"

namespace emida
{

template<typename T>
class gpu_offset
{
private:
	T* cu_pic_;
	T* cu_temp_;
	T* cu_sums_pic_;
	T* cu_sums_temp_;
	T* cu_hann_x_;
	T* cu_hann_y_;
	T* cu_cross_res_;
	T* cu_neighbors_;
	size2_t* cu_maxes_i_;
	data_index<T>* cu_maxes_;

	vec2<size_t> size_;
	vec2<size_t> cross_size_;
	size_t b_size_;
	size_t one_size_;
	size_t neigh_size_;
	size_t maxarg_block_size_ = 1024;
	size_t maxarg_one_pic_blocks_;
	size_t maxarg_maxes_size_;

	static constexpr int s = 3;
	static constexpr int r = (s - 1) / 2;
public:
	gpu_offset(vec2<size_t> size, vec2<size_t> cross_size, size_t b_size)
		: size_(size)
		, cross_size_(cross_size)
		, b_size_(b_size)
		, one_size_(size_.area())
		, neigh_size_(s * s * b_size)
		, maxarg_one_pic_blocks_(div_up(cross_size.area(), maxarg_block_size_))
		, maxarg_maxes_size_(maxarg_one_pic_blocks_* b_size)
	{}

	void allocate_memory()
	{
		cu_pic_ = cuda_malloc<T>(one_size_ * b_size_);
		cu_temp_ = cuda_malloc<T>(one_size_ * b_size_);

		cu_sums_pic_ = cuda_malloc<T>(b_size_);
		cu_sums_temp_ = cuda_malloc<T>(b_size_);

		auto hann_x = generate_hanning<T>(size_.x);
		auto hann_y = generate_hanning<T>(size_.y);

		cu_hann_x_ = vector_to_device(hann_x);
		cu_hann_y_ = vector_to_device(hann_y);

		cu_cross_res_ = cuda_malloc<T>(cross_size_.area() * b_size_);

		cu_maxes_ = cuda_malloc<data_index<T>>(maxarg_maxes_size_);


		if (!(cross_size_.x == s && cross_size_.y == s))
		{
			cu_neighbors_ = cuda_malloc<T>(neigh_size_);

			cu_maxes_i_ = cuda_malloc<size2_t>(b_size_);
		}

	}

	//gets two pictures with size cols x rows and returns subpixel offset between them
	std::vector<vec2<T>> get_offset(T* pic, T* temp) const
	{
		stopwatch sw(true, 2, 2);
		
		copy_to_device(pic, one_size_ * b_size_, cu_pic_);
		copy_to_device(temp, one_size_ * b_size_, cu_temp_); sw.tick("Temp and pic to device: ");

		run_sum(cu_pic_, cu_sums_pic_, one_size_, b_size_);
		run_sum(cu_temp_, cu_sums_temp_, one_size_, b_size_);

		CUCH(cudaGetLastError());
		CUCH(cudaDeviceSynchronize()); sw.tick("Run sums: ");

		run_prepare_pics(cu_pic_, cu_hann_x_, cu_hann_y_, cu_sums_pic_, size_, b_size_);
		run_prepare_pics(cu_temp_, cu_hann_x_, cu_hann_y_, cu_sums_temp_, size_, b_size_);

		CUCH(cudaGetLastError());
		CUCH(cudaDeviceSynchronize()); sw.tick("Run prepare: ");

		run_cross_corr(cu_temp_, cu_pic_, cu_cross_res_, size_, cross_size_, b_size_);

		CUCH(cudaGetLastError());
		CUCH(cudaDeviceSynchronize()); sw.tick("Run cross corr: ");

		T* cu_neighbors = cu_neighbors_;
		std::vector<vec2<size_t>> maxes_i;
		if (cross_size_.x == s && cross_size_.y == s)
		{
			cu_neighbors = cu_cross_res_;
		}
		else
		{
			maxes_i = get_maxarg(cu_cross_res_); sw.tick("Get maxarg: ");

			copy_to_device(maxes_i, cu_maxes_i_); sw.tick("Maxes transfer: ");

			run_extract_neighbors<T, s>(cu_cross_res_, cu_maxes_i_, cu_neighbors, cross_size_.x, cross_size_.y, b_size_);

			CUCH(cudaGetLastError());
			CUCH(cudaDeviceSynchronize()); sw.tick("Run extract neigh: ");
		}

		std::vector<T> neighbors = device_to_vector(cu_neighbors, neigh_size_); sw.tick("Transfer neighbors: ");

		auto subp_offset = subpixel_max_serial<T, s>(neighbors.data(), b_size_); sw.tick("Subpixel max: ");

		std::vector<vec2<T>> res(b_size_);
		if (cross_size_.x == s && cross_size_.y == s)
		{
			for (size_t i = 0; i < b_size_; ++i)
			{
				res[i] = subp_offset[i] - r;
			}
		}
		else
		{
			for (size_t i = 0; i < b_size_; ++i)
			{

				res[i].x = (int)maxes_i[i].x - ((int)cross_size_.x / 2) - r + subp_offset[i].x;
				res[i].y = (int)maxes_i[i].y - ((int)cross_size_.y / 2) - r + subp_offset[i].y;
			}
		}sw.tick("Offsets finalisation: ");
		sw.total();
		return res;
	}

	inline std::vector<vec2<size_t>> get_maxarg(const T* cu_data) const
	{
		run_maxarg_reduce(cu_data, cu_maxes_, cross_size_.area(), maxarg_block_size_, b_size_);

		CUCH(cudaGetLastError());
		CUCH(cudaDeviceSynchronize());

		std::vector<data_index<T>> maxes = device_to_vector(cu_maxes_, maxarg_maxes_size_);

		std::vector<vec2<size_t>> res(b_size_);

		for (size_t b = 0; b < b_size_; ++b)
		{
			size_t max_res_i = b * maxarg_one_pic_blocks_;
			for (size_t i = max_res_i + 1; i < (b + 1) * maxarg_one_pic_blocks_; ++i)
			{
				if (maxes[i].data > maxes[max_res_i].data)
					max_res_i = i;
			}
			size_t max_i = maxes[max_res_i].index - b * cross_size_.area();
			res[b].x = max_i % cross_size_.x;
			res[b].y = max_i / cross_size_.x;
		}

		return res;
	}
};

}