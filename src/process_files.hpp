#pragma once
#include <filesystem>
#include <fstream>
#include <thread>
#include <condition_variable>
#include <mutex>
#include <atomic>

#include "get_offset.hpp"
#include "load_tiff.hpp"
#include "slice_picture.hpp"
#include "draw_tiff.hpp"
#include "args_parser.hpp"
#include "concurrent_queue.hpp"

namespace emida
{

struct file
{
	double x;
	double y;
	std::string fname;
};

struct offs_job
{
	esize_t batch_files;
	esize_t c;
};

template<typename T>
struct fin_job
{
	data_index<T> * maxes_i;
	T * neighbors;
	esize_t batch_files;
	esize_t c;
};

template<typename T>
class file_processor
{
	std::vector<file> load_work(std::string fname)
	{
		std::vector<file> res;
		std::ifstream infile(fname);
		std::string line;

		while (std::getline(infile, line))
		{
			file f;
			std::stringstream iss(line);
		
		
			iss >> f.x;
			iss >> f.y;
			iss.ignore();
			std::getline(iss, f.fname);
			res.push_back(std::move(f));
		}
		return res;
	}

	params a;
	size2_t pic_size;
	std::vector<size2_t> slice_begins;
	std::vector<uint16_t> initial_raster;
	gpu_offset<T, uint16_t> offs;
	std::vector<file> work;
	stopwatch sw;

	fin_job<T> fin_job1;
	fin_job<T> fin_job2;

	concurrent_queue<offs_job> load_queue;
	concurrent_queue<fin_job<T>*> fin_queue;

public:
	file_processor(params par) : a(std::move(par)), sw(true, 3)
	{
		slice_begins = a.slice_mids;
		for (auto& o : slice_begins)
			o = o - (a.slice_size / 2);
	}


	inline void process_files()
	{
		

		if (!get_size(a.initial_file_name, pic_size))
			return;
	
		initial_raster.resize(pic_size.area());

		if (!load_tiff(a.initial_file_name, initial_raster.data(), pic_size))
			return;
	
		offs = gpu_offset<T, uint16_t>(pic_size, &slice_begins, a.slice_size, a.cross_size, a.batch_size, a.cross_pol, a.fitting_size);
		offs.allocate_memory(initial_raster.data());

		uint16_t* deformed_raster1 = cuda_malloc_host<uint16_t>(pic_size.area() * a.batch_size);
		uint16_t* deformed_raster2 = cuda_malloc_host<uint16_t>(pic_size.area() * a.batch_size);

		uint16_t* deformed_raster = deformed_raster1;
		uint16_t* deformed_raster_next = deformed_raster2;

		esize_t total_slices = (esize_t)a.slice_mids.size() * a.batch_size;

		fin_job1.maxes_i = cuda_malloc_host<data_index<T>>(total_slices);
		fin_job2.maxes_i = cuda_malloc_host<data_index<T>>(total_slices);

		fin_job1.neighbors = cuda_malloc_host<T>(total_slices * a.fitting_size * a.fitting_size);
		fin_job2.neighbors = cuda_malloc_host<T>(total_slices * a.fitting_size * a.fitting_size);

		work = load_work(a.deformed_list_file_name);
		
		if (work.size() == 0)
		{
			std::cerr << "The deformed list is empty.\n";
			return;
		}

		std::thread comp_offs_thr(&file_processor::compute_offsets_thread, this);
		std::thread finalize_thr(&file_processor::finalize_thread, this);
		sw.zero();
		for (esize_t c = 0; c < (esize_t)work.size(); c += a.batch_size)
		{
			esize_t batch_files = c + a.batch_size > (esize_t)work.size() ? (esize_t)work.size() - c : a.batch_size;
			bool OK = true;
			stopwatch swa;
			uint16_t* next = deformed_raster;
			swa.zero();
			for (esize_t i = c; i < c + batch_files; ++i)
			{
				OK &= load_tiff(work[i].fname, next, pic_size);
				next += pic_size.area();
			}//swa.tock("Load tiff: ");

			if (!OK)
				continue;
			
			//swa.zero();
			//auto v = get_pics<uint16_t>(job->deformed_raster, pic_size, slice_begins, a.slice_size);
			//swa.tick("GET PICS TEST BLA");

			offs.transfer_pic_to_device_async(deformed_raster);

			offs_job job = { batch_files, c };
			
			load_queue.push(job);
			
			std::swap(deformed_raster, deformed_raster_next);
			

			//sw.tick("ONE: ", 1);
		}
		comp_offs_thr.join();
		finalize_thr.join();

		sw.total();
	}

	void compute_offsets_thread()
	{
		fin_job<T>* fjob = &fin_job1;
		fin_job<T>* fjob_next = &fin_job2;
		for (;;)
		{
			load_queue.wait_for_data();

			cudaStreamSynchronize(offs.in_stream);

			auto job = load_queue.front();
			
			offs.get_offset_core(fjob->maxes_i, fjob->neighbors);

			load_queue.pop();

			fjob->c = job.c;
			fjob->batch_files = job.batch_files;

			fin_queue.push(fjob);
			
			if (fjob->c + a.batch_size >= work.size())
				break;

			std::swap(fjob, fjob_next);

			
		}
	}

	void finalize_thread()
	{
		for (;;)
		{
			fin_queue.wait_for_data();
			
			finalize_offsets(*fin_queue.front());

			if (fin_queue.front()->c + a.batch_size >= work.size())
				break;
			fin_queue.pop();
		}
	}

	void finalize_offsets(fin_job<T>& fjob)
	{
		auto [offsets, coefs] = offs.finalize(fjob.maxes_i, fjob.neighbors);
		//sw.tick("Get offset: ", 2);

		if (a.analysis)
			stopwatch::global_stats.inc_histogram(offsets);
		for (esize_t j = fjob.c; j < fjob.c + fjob.batch_files; ++j)
		{
			printf("%f %f %llu %s\n", work[j].x, work[j].y, a.slice_mids.size(), work[j].fname.c_str());
			esize_t it = j - fjob.c;
			for (esize_t i = it * (esize_t)a.slice_mids.size(); i < (it + 1) * (esize_t)a.slice_mids.size(); ++i)
			{
				esize_t begin_i = i % (esize_t)a.slice_mids.size();
				//sometimes the resulting float is outputted as nan and sometimes as nan(ind). Normalize that here.
				if (std::isnan(offsets[i].x))
					offsets[i].x = std::numeric_limits<T>::quiet_NaN();
				if (std::isnan(offsets[i].y))
					offsets[i].y = std::numeric_limits<T>::quiet_NaN();
				if (a.write_coefs)
					printf("%lu %lu %f %f %f %f %f %f %f %f\n",
						a.slice_mids[begin_i].x, a.slice_mids[begin_i].y,
						offsets[i].x, offsets[i].y,
						coefs[i][0], coefs[i][1], coefs[i][2], coefs[i][3], coefs[i][4], coefs[i][5]);
				else
					printf("%lu %lu %f %f\n", a.slice_mids[begin_i].x, a.slice_mids[begin_i].y, offsets[i].x, offsets[i].y);
			}

		}
	}

};




template<typename T>
void write_offsets(const params& a, std::vector<vec2<T>>& offsets, const std::vector<std::array<T, 6>>& coefs)
{
	for (esize_t i = 0; i < offsets.size(); ++i)
	{
		//sometimes the resulting float is outputted as nan and sometimes as nan(ind). Normalize that here.
		if (std::isnan(offsets[i].x))
			offsets[i].x = std::numeric_limits<T>::quiet_NaN();
		if (std::isnan(offsets[i].y))
			offsets[i].y = std::numeric_limits<T>::quiet_NaN();
		if (a.write_coefs)
			printf("%llu %llu %f %f %f %f %f %f %f %f\n", a.slice_mids[i].x, a.slice_mids[i].y, offsets[i].x, offsets[i].y,
				coefs[i][0], coefs[i][1], coefs[i][2], coefs[i][3], coefs[i][4], coefs[i][5]);
		else
			printf("%llu %llu %f %f\n", a.slice_mids[i].x, a.slice_mids[i].y, offsets[i].x, offsets[i].y);
	}
}

}
