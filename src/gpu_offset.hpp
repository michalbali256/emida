#pragma once

#include "host_helpers.hpp"
#include "subpixel_max.hpp"
#include "kernels.cuh"
#include "subtract_mean.hpp"
#include "stopwatch.hpp"
#include "slice_picture.hpp"
#include "cufft_helpers.hpp"

namespace emida
{

template<typename T, typename IN>
class gpu_offset
{
private:
	IN* cu_pic_in_;
	IN* cu_temp_in_;
	T* cu_pic_;
	T* cu_temp_;
	T* cu_sums_pic_;
	T* cu_sums_temp_;
	T* cu_hann_x_;
	T* cu_hann_y_;
	T* cu_cross_res_;
	T* cu_neighbors_;
	size2_t* cu_maxes_i_;
	size2_t* cu_begins_;
	data_index<T>* cu_maxes_;

	size2_t src_size_;
	const std::vector<size2_t>* begins_;
	size2_t slice_size_;
	size2_t cross_in_size_;
	size2_t cross_size_;
	size_t b_size_;
	size_t neigh_size_;
	size_t maxarg_block_size_ = 1024;
	size_t maxarg_one_pic_blocks_;
	size_t maxarg_maxes_size_;
	int s;
	int r;

	int fft_size_[2];
	cufftHandle plan_;
	cufftHandle inv_plan_;

	cross_policy cross_policy_;

	mutable stopwatch sw;
public:
	gpu_offset(size2_t src_size, const std::vector<size2_t>* begins, size2_t slice_size, size2_t cross_size, cross_policy policy = CROSS_POLICY_BRUTE, int s = 3)
		: src_size_(src_size)
		, begins_(begins)
		, slice_size_(slice_size)
		, cross_in_size_(policy == CROSS_POLICY_BRUTE ? slice_size : size2_t{slice_size.x * 2 + 2, slice_size.y * 2})
		, cross_size_(policy == CROSS_POLICY_BRUTE ? cross_size : slice_size * 2 - 1)
		, b_size_(begins->size())
		, neigh_size_(s * s * b_size_)
		, maxarg_one_pic_blocks_(div_up(cross_size_.area(), maxarg_block_size_))
		, maxarg_maxes_size_(maxarg_one_pic_blocks_* b_size_)
		, sw(false
			, 2, 2)
		, s(s)
		, r((s - 1) / 2)
		, fft_size_{ (int)slice_size.x * 2, (int)slice_size.y * 2 }
		, cross_policy_(policy)
	{}

	void allocate_memory(IN* temp)
	{
		cu_pic_in_ = cuda_malloc<IN>(src_size_.area());
		cu_temp_in_ = cuda_malloc<IN>(src_size_.area());
		
		//Enforce alignment by allocating cufft complex types
		//Result we also need to allocate one row more because that is the size of ff-transpformed result
		cu_pic_ = (T*) cuda_malloc<typename complex_trait<T>::type>(cross_in_size_.area() / 2 * b_size_);
		cu_temp_ = (T*) cuda_malloc<typename complex_trait<T>::type>(cross_in_size_.area() / 2 * b_size_);

		cu_sums_pic_ = cuda_malloc<T>(b_size_);
		cu_sums_temp_ = cuda_malloc<T>(b_size_);

		cu_begins_ = vector_to_device(*begins_);

		auto hann_x = generate_hanning<T>(slice_size_.x);
		auto hann_y = generate_hanning<T>(slice_size_.y);

		cu_hann_x_ = vector_to_device(hann_x);
		cu_hann_y_ = vector_to_device(hann_y);

		//cu_cross_res_ = cross_policy_ == CROSS_POLICY_FFT ? cu_pic_ : cuda_malloc<T>(cross_size_.area() * b_size_);
		cu_cross_res_ = cuda_malloc<T>(cross_size_.area() * b_size_);

		cu_maxes_ = cuda_malloc<data_index<T>>(maxarg_maxes_size_);


		if (!(cross_size_.x == s && cross_size_.y == s))
		{
			cu_neighbors_ = cuda_malloc<T>(neigh_size_);

			cu_maxes_i_ = cuda_malloc<size2_t>(b_size_);
		}

		FFTCH(cufftPlanMany(&plan_, 2, fft_size_,
			NULL, 1, 0,
			NULL, 1, 0,
			fft_type_R2C<T>(), (int)b_size_));


		FFTCH(cufftPlanMany(&inv_plan_, 2, fft_size_,
			NULL, 1, 0,
			NULL, 1, 0,
			fft_type_C2R<T>(), (int)b_size_));

		//prepare temp

		if (temp == nullptr)
			return;
		copy_to_device(temp, src_size_.area(), cu_temp_in_);

		run_sum(cu_temp_in_, cu_sums_temp_, cu_begins_, src_size_, slice_size_, b_size_);

		run_prepare_pics(cu_temp_in_, cu_temp_, cu_hann_x_, cu_hann_y_, cu_sums_temp_, cu_begins_, src_size_, slice_size_, cross_in_size_, b_size_);

		if(cross_policy_ == CROSS_POLICY_FFT)
			fft_real_to_complex(plan_, cu_temp_);
	}

	//gets two pictures with size cols x rows and returns subpixel offset between them
	offsets_t<T> get_offset(IN* pic) const
	{
		sw.zero();
		
		copy_to_device(pic, src_size_.area(), cu_pic_in_);
		sw.tick("Pic to device: ");

		run_sum(cu_pic_in_, cu_sums_pic_, cu_begins_, src_size_, slice_size_, b_size_);

		CUCH(cudaGetLastError());
		CUCH(cudaDeviceSynchronize()); sw.tick("Run sums: ");

		run_prepare_pics(cu_pic_in_, cu_pic_, cu_hann_x_, cu_hann_y_, cu_sums_pic_, cu_begins_, src_size_, slice_size_, cross_in_size_, b_size_);

		CUCH(cudaGetLastError());
		CUCH(cudaDeviceSynchronize()); sw.tick("Run prepare: ");

		if(cross_policy_ == CROSS_POLICY_BRUTE)
			run_cross_corr(cu_pic_, cu_temp_, cu_cross_res_, slice_size_, cross_size_, b_size_);
		else
			cross_corr_fft();

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

			run_extract_neighbors<T>(cu_cross_res_, cu_maxes_i_, cu_neighbors, s, cross_size_, b_size_);

			CUCH(cudaGetLastError());
			CUCH(cudaDeviceSynchronize()); sw.tick("Run extract neigh: ");
		}

		std::vector<T> neighbors = device_to_vector(cu_neighbors, neigh_size_); sw.tick("Transfer neighbors: ");

		auto [subp_offset, coefs] = subpixel_max_serial<T>(neighbors.data(), s, b_size_); sw.tick("Subpixel max: ");

		std::vector<vec2<T>> res(b_size_);
		if (cross_size_.x == s && cross_size_.y == s)
		{
			for (size_t i = 0; i < b_size_; ++i)
			{
				res[i] = -(subp_offset[i] - r);
			}
		}
		else
		{
			for (size_t i = 0; i < b_size_; ++i)
			{

				res[i].x =-((int)maxes_i[i].x - ((int)cross_size_.x / 2) - r + subp_offset[i].x);
				res[i].y =-((int)maxes_i[i].y - ((int)cross_size_.y / 2) - r + subp_offset[i].y);
			}
		}sw.tick("Offsets finalisation: ");
		sw.total();
		return { res, coefs };
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
			++stopwatch::global_stats.total_pics;
		}

		return res;
	}

	
	inline void cross_corr_fft() const
	{
		stopwatch sw(true, 2, 4);

		//auto vec = device_to_vector(cu_pic_, cross_in_size_.area() * b_size_);
		CUCH(cudaDeviceSynchronize());

		

		fft_real_to_complex(plan_, cu_pic_);
		CUCH(cudaDeviceSynchronize()); sw.tick("R2C: ");

		//auto pppic = device_to_vector(cu_pic_, cross_in_size_.area() * b_size_);
		//auto temp = device_to_vector(cu_temp_, cross_in_size_.area() * b_size_);

		run_hadamard((typename complex_trait<T>::type*)cu_pic_, (typename complex_trait<T>::type*)cu_temp_, { cross_in_size_.x / 2, cross_in_size_.y }, b_size_);
		CUCH(cudaDeviceSynchronize()); sw.tick("Multiply: ");

		
		//auto vec2 = device_to_vector(cu_pic_, cross_in_size_.area() * b_size_);

		//auto out = cuda_malloc<T>(cross_in_size_.area());

		fft_complex_to_real(inv_plan_, cu_pic_);
		CUCH(cudaDeviceSynchronize()); sw.tick("C2R: ");

		run_finalize_fft(cu_pic_, cu_cross_res_, cross_size_, b_size_);
		CUCH(cudaDeviceSynchronize()); sw.tick("finalize: ");

		//auto vec3 = device_to_vector(cu_pic_, cross_in_size_.area() * b_size_);
		//auto vec4 = device_to_vector(cu_cross_res_, cross_size_.area() * b_size_);

		int i = 0;
	}

	T* get_cu_pic()
	{
		return cu_pic_;
	}
	T* get_cu_temp()
	{
		return cu_temp_;
	}
};

}