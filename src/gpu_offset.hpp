#pragma once

#include<atomic>

#include "host_helpers.hpp"
#include "subpixel_max.hpp"
#include "kernels.cuh"
#include "subtract_mean.hpp"
#include "stopwatch.hpp"
#include "slice_picture.hpp"
#include "cufft_helpers.hpp"

// Uncomment to get the measurements of individual kernels
//#define MEASURE_KERNELS

#ifdef MEASURE_KERNELS
#	define TICK(sw, label) CUCH(cudaGetLastError()); CUCH(cudaDeviceSynchronize()); sw.tick(label);
#else
#	define TICK(sw, label)
#endif

namespace emida
{

// Class, that implements the pattern processing algorithm on gpu by using the kernels from kernels.cuh, allocating memory etc.
// Processes one batch (several files) at one call.


// The most important functions are:
// allocate_memory: it is called only once to allocate all the gpu buffers
// transfer_pic_to_device_async:
//     copies input pattern(s) from host to gpu memory. It is done asynchronously and can be used by
//     several load workers at the same time. That is why it uses several input buffers. We use atomic integer to determine the
//     index of next buffer to use, so two workers never copy data to the same buffer at the same time+
// gpu_offset_core:
//     Processes one batch of of input patterns on the gpu. Ends by copying the positions of maxima and their neighborhoods from
//     the GPU to the host memory
// finalize:
//     Processes the maxima and their neighborhoods to get final offsets. Executed on the CPU.

// All the buffers use row-major order of the picters and pixels. In buffers that contain more subregions, the subregions are
// placed immediately one after another.
// The same applies for subregions of different patterns when batch_size>1
// For kernels that operate with the position of subregions, the number of subregions as well as batch_size has to be passed
// to the kernel, since they have to use the subregion description of each pattern in the batch separately (batch_size-times)


template<typename T, typename IN>
class gpu_offset
{
private:
	using complex_t = typename complex_trait<T>::type;
	using sums_t = typename sums_trait<IN>::type;

	std::vector<IN*> cu_in_buffers_;
	std::atomic<esize_t> in_buffers_i_ = 0;

	IN* cu_temp_in_;
	T* cu_pic_;
	T* cu_temp_;
	sums_t* cu_sums_pic_;
	sums_t* cu_sums_temp_;
	T* cu_hann_x_;
	T* cu_hann_y_;
	T* cu_cross_res_;
	T* cu_fft_res_;
	T* cu_neighbors_;
	data_index<T>* cu_maxes_i_;
	size2_t* cu_begins_;
	data_index<T>* cu_maxes_;

	size2_t src_size_;
	const std::vector<size2_t>* begins_;
	size2_t slice_size_;
	size2_t cross_in_size_;
	size2_t fft_work_size_;
	size2_t cross_size_;
	esize_t total_slices_;
	esize_t batch_size_;
	esize_t neigh_size_;
	esize_t maxarg_block_size_ = 1024;
	esize_t maxarg_one_pic_blocks_;
	esize_t maxarg_maxes_size_;
	int s;
	int r;

	int fft_size_[2];
	cufftHandle plan_;
	cufftHandle inv_plan_;

	cross_policy cross_policy_;

	mutable stopwatch sw;
public:

	cudaStream_t in_stream;
	cudaStream_t out_stream;

	gpu_offset() {}
	gpu_offset(size2_t src_size,
		const std::vector<size2_t>* begins,
		size2_t slice_size,
		size2_t cross_size,
		esize_t batch_size,
		cross_policy policy = CROSS_POLICY_BRUTE,
		int s = 3,
		size_t in_buffers_size = 1)
		: cu_in_buffers_(in_buffers_size)
		, src_size_(src_size)
		, begins_(begins)
		, slice_size_(slice_size)
		, cross_in_size_(policy == CROSS_POLICY_BRUTE ? slice_size : size2_t{ slice_size.x * 2, slice_size.y * 2 })
		, fft_work_size_(policy == CROSS_POLICY_BRUTE ? slice_size : size2_t{ slice_size.x * 2 + 2, slice_size.y * 2 })
		, cross_size_(policy == CROSS_POLICY_BRUTE ? cross_size : slice_size * 2 - 1)
		, total_slices_((esize_t)begins->size()* batch_size)
		, batch_size_(batch_size)
		, neigh_size_(s* s* total_slices_)
		, maxarg_one_pic_blocks_(div_up(cross_size_.area(), maxarg_block_size_))
		, maxarg_maxes_size_(maxarg_one_pic_blocks_* total_slices_)
		, sw(true, 2, 2)
		, s(s)
		, r((s - 1) / 2)
		, fft_size_{ (int)slice_size.x * 2, (int)slice_size.y * 2 }
		, cross_policy_(policy)
	{}

	void allocate_memory(IN* temp)
	{
		CUCH(cudaStreamCreate(&in_stream));
		CUCH(cudaStreamCreate(&out_stream));

		for (auto& buf : cu_in_buffers_)
			buf = cuda_malloc<IN>(src_size_.area() * batch_size_);
		cu_temp_in_ = cuda_malloc<IN>(src_size_.area());

		//Enforce alignment by allocating cufft complex types
		//Result we also need to allocate one row more because that is the size of ff-transpformed result
		cu_pic_ = (T*)cuda_malloc<complex_t>(cross_in_size_.area() / 2 * total_slices_);
		cu_temp_ = (T*)cuda_malloc<complex_t>(fft_work_size_.area() / 2 * begins_->size());



		cu_sums_pic_ = cuda_malloc<sums_t>(total_slices_);
		cu_sums_temp_ = cuda_malloc<sums_t>(begins_->size());

		cu_begins_ = vector_to_device(*begins_);

		auto hann_x = generate_hanning<T>(slice_size_.x);
		auto hann_y = generate_hanning<T>(slice_size_.y);

		cu_hann_x_ = vector_to_device(hann_x);
		cu_hann_y_ = vector_to_device(hann_y);

		if (cross_policy_ == cross_policy::CROSS_POLICY_BRUTE)
			cu_cross_res_ = cuda_malloc<T>(cross_size_.area() * total_slices_);
		else
		{
			cu_fft_res_ = (T*)cuda_malloc<complex_t>(fft_work_size_.area() / 2 * total_slices_);
			cu_cross_res_ = (T*)cuda_malloc<complex_t>(cross_in_size_.area() / 2 * total_slices_);
			cuda_memset(cu_pic_, 0, cross_in_size_.area() * total_slices_);
		}

		cu_maxes_ = cuda_malloc<data_index<T>>(maxarg_maxes_size_);


		cu_neighbors_ = cuda_malloc<T>(neigh_size_);

		cu_maxes_i_ = cuda_malloc<data_index<T>>(total_slices_);


		if (cross_policy_ == CROSS_POLICY_FFT)
		{
			FFTCH(cufftPlanMany(&plan_, 2, fft_size_,
				NULL, 1, 0,
				NULL, 1, 0,
				fft_type_R2C<T>(), (int)total_slices_));


			FFTCH(cufftPlanMany(&inv_plan_, 2, fft_size_,
				NULL, 1, 0,
				NULL, 1, 0,
				fft_type_C2R<T>(), (int)total_slices_));
		}

		//prepare temp

		if (temp == nullptr)
			return;
		copy_to_device(temp, src_size_.area(), cu_temp_in_);

		run_sum(cu_temp_in_, cu_sums_temp_, cu_begins_, src_size_, slice_size_, (esize_t)begins_->size(), 1);
		CUCH(cudaDeviceSynchronize());

		run_prepare_pics(cu_temp_in_, cu_temp_,
			cu_hann_x_, cu_hann_y_, cu_sums_temp_, cu_begins_,
			src_size_, slice_size_, fft_work_size_, (esize_t)begins_->size(), 1);
		CUCH(cudaDeviceSynchronize());

		if (cross_policy_ == CROSS_POLICY_FFT)
		{
			cufftHandle temp_plan;
			FFTCH(cufftPlanMany(&temp_plan, 2, fft_size_,
				NULL, 1, 0,
				NULL, 1, 0,
				fft_type_R2C<T>(), (int)begins_->size()));

			fft_real_to_complex(temp_plan, cu_temp_);

			FFTCH(cufftDestroy(temp_plan));
		}
	}

	esize_t transfer_pic_to_device_async(IN* pic)
	{
		auto i = in_buffers_i_.fetch_add(1) % (esize_t)cu_in_buffers_.size();
		copy_to_device_async(pic, src_size_.area() * batch_size_, cu_in_buffers_[i], in_stream);
		return i;
	}

	offsets_t<double> get_offset(IN* pic)
	{
		sw.zero();
		copy_to_device(pic, src_size_.area() * batch_size_, cu_in_buffers_[0]);
		sw.tick("Pic to device: ");

		std::vector<data_index<T>> maxes_i(total_slices_);
		std::vector<T> neighbors(neigh_size_);
		get_offset_core(maxes_i.data(), neighbors.data(), 0);

		return finalize(maxes_i.data(), neighbors.data());
	}

	//gets two pictures with size cols x rows and returns subpixel offset between them
	void get_offset_core(data_index<T>* maxes_i, T* neighbors, size_t cu_pic_index)
	{
		sw.zero();
		run_sum(cu_in_buffers_[cu_pic_index], cu_sums_pic_, cu_begins_, src_size_, slice_size_, (esize_t)begins_->size(), batch_size_);

		TICK(sw, "Run sums: ")

		run_prepare_pics(cu_in_buffers_[cu_pic_index], cu_pic_,
			cu_hann_x_, cu_hann_y_, cu_sums_pic_, cu_begins_,
			src_size_, slice_size_, cross_in_size_, (esize_t)begins_->size(), batch_size_);

		TICK(sw, "Run prepare: ")

		if (cross_policy_ == CROSS_POLICY_BRUTE)
			run_cross_corr(cu_pic_, cu_temp_, cu_cross_res_, slice_size_, cross_size_, (esize_t)begins_->size(), batch_size_);
		else
			cross_corr_fft();

		TICK(sw, "Run cross corr: ")

		T* cu_neighbors = cu_neighbors_;

		CUCH(cudaMemset(cu_maxes_i_, 0, total_slices_ * sizeof(data_index<T>)));
		if (cross_policy_ == CROSS_POLICY_BRUTE)
			run_maxarg_reduce(cu_cross_res_, cu_maxes_, cu_maxes_i_, cross_size_, maxarg_block_size_, total_slices_);
		else
			run_maxarg_reduce<T>(cu_cross_res_, cu_maxes_, cu_maxes_i_, { cross_size_.x + 1, cross_size_.y + 1 }, maxarg_block_size_, total_slices_);

		CUCH(cudaDeviceSynchronize());
		TICK(sw, "Run maxarg: ")

		copy_from_device_async<data_index<T>>(cu_maxes_i_, maxes_i, total_slices_, out_stream);

		sw.zero();

		if (cross_policy_ == CROSS_POLICY_BRUTE)
			run_extract_neighbors<T>(cu_cross_res_, cu_maxes_i_, cu_neighbors, s, cross_size_, total_slices_);
		else
			run_extract_neighbors<T, cross_res_pos_policy_fft>(cu_cross_res_, cu_maxes_i_, cu_neighbors, s, cross_size_, total_slices_);

		CUCH(cudaDeviceSynchronize());
		TICK(sw, "Run extract neigh: ")

			copy_from_device_async(cu_neighbors, neighbors, neigh_size_, out_stream); sw.tick("Transfer neighbors: ");
	}

	offsets_t<double> finalize(data_index<T>* maxes_i, T* neighbors)
	{
		CUCH(cudaStreamSynchronize(out_stream));
		stopwatch swf;
		swf.zero();
		auto [subp_offset, coefs] = subpixel_max_serial<T>(neighbors, s, total_slices_); swf.tick("Subpixel max: ");

		std::vector<vec2<double>> res(total_slices_);

		for (esize_t i = 0; i < total_slices_; ++i)
		{
			size2_t max_pos;
			if (cross_policy_ == cross_policy::CROSS_POLICY_BRUTE)
				max_pos = cross_res_pos_policy_id::shift_pos(maxes_i[i].index, cross_size_);
			else
				max_pos = cross_res_pos_policy_fft::shift_pos(maxes_i[i].index, cross_size_);
			res[i].x = -((int)max_pos.x - ((int)cross_size_.x / 2) - r + subp_offset[i].x);
			res[i].y = -((int)max_pos.y - ((int)cross_size_.y / 2) - r + subp_offset[i].y);
		}
		swf.tick("Offsets finalisation: ");
		return { res, coefs };
	}

	inline void cross_corr_fft() const
	{
		fft_real_to_complex(plan_, cu_pic_, (complex_t*)cu_fft_res_);
		TICK(sw, "R2C: ")


		run_hadamard((complex_t*)cu_fft_res_, (complex_t*)cu_temp_,
			{ fft_work_size_.x / 2, fft_work_size_.y },
			(esize_t)begins_->size(), batch_size_);
		TICK(sw, "Multiply: ");

		fft_complex_to_real(inv_plan_, (complex_t*)cu_fft_res_, cu_cross_res_);
		TICK(sw, "C2R: ");
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