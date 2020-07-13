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
	std::vector<uint16_t> deformed_raster;
	std::atomic<bool> loaded = false;
	size_t batch_files;
	size_t c;
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
	std::mutex mtx;
	std::condition_variable cond_compute_offs;
	
	offs_job job1;
	offs_job job2;


	std::atomic<bool> offs_done = true;
	std::atomic<bool> end_comp_offs = false;

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

		
		job1.deformed_raster.resize(pic_size.area() * a.batch_size);
		job2.deformed_raster.resize(pic_size.area() * a.batch_size);
		offs_job* job = &job1;
		offs_job* job_next = &job2;

		work = load_work(a.deformed_list_file_name);
	
		std::thread comp_offs(&file_processor::compute_offsets_thread, this);
		sw.zero();
		for (size_t c = 0; c < work.size(); c += a.batch_size)
		{
			size_t batch_files = c + a.batch_size > work.size() ? work.size() - c : a.batch_size;
			bool OK = true;
			uint16_t * next = job->deformed_raster.data();
			for (size_t i = c; i < c + batch_files; ++i)
			{
				OK &= load_tiff(work[i].fname, next, pic_size);
				next += pic_size.area();
			}sw.tick("Load tiff: ", 2);

			if (!OK)
				continue;

			std::unique_lock lck(mtx);
			job->batch_files = batch_files;
			job->c = c;
			job->loaded = true;
			
			//compute_offsets(*job);
			cond_compute_offs.wait(lck, [&]() {return offs_done.load(); });
			std::swap(job, job_next);
			offs_done = false;

			sw.tick("ONE: ", 1);
		}
		end_comp_offs = true;
		comp_offs.join();

		sw.total();
		if (a.analysis)
		{
			std::cerr << "Total pics: " << stopwatch::global_stats.total_pics << "\n";
			stopwatch::global_stats.write_histogram();
		}
	}

	void compute_offsets_thread()
	{
		offs_job* job = &job1;
		offs_job* job_next = &job2;

		for (;;)
		{
			while (!job->loaded && !end_comp_offs);
			if (end_comp_offs)
				return;

			cudaStreamSynchronize(offs.in_stream);
			compute_offsets(*job);
			offs_done = true;
			job->loaded = false;
			cond_compute_offs.notify_one();
			std::swap(job, job_next);
		}
	}

	void compute_offsets(offs_job& job)
	{
		offs.get_offset_core();
		auto [offsets, coefs] = offs.finalize();
		
		
		//sw.tick("Get offset: ", 2);

		if (a.analysis)
			stopwatch::global_stats.inc_histogram(offsets);
		for (size_t j = job.c; j < job.c + job.batch_files; ++j)
		{
			printf("%f %f %llu %s\n", work[j].x, work[j].y, a.slice_mids.size(), work[j].fname.c_str());
			size_t it = j - job.c;
			for (size_t i = it * a.slice_mids.size(); i < (it + 1) * a.slice_mids.size(); ++i)
			{
				size_t begin_i = i % a.slice_mids.size();
				//sometimes the resulting float is outputted as nan and sometimes as nan(ind). Normalize that here.
				if (std::isnan(offsets[i].x))
					offsets[i].x = std::numeric_limits<T>::quiet_NaN();
				if (std::isnan(offsets[i].y))
					offsets[i].y = std::numeric_limits<T>::quiet_NaN();
				if (a.write_coefs)
					printf("%llu %llu %f %f %f %f %f %f %f %f\n",
						a.slice_mids[begin_i].x, a.slice_mids[begin_i].y,
						offsets[i].x, offsets[i].y,
						coefs[i][0], coefs[i][1], coefs[i][2], coefs[i][3], coefs[i][4], coefs[i][5]);
				else
					printf("%llu %llu %f %f\n", a.slice_mids[begin_i].x, a.slice_mids[begin_i].y, offsets[i].x, offsets[i].y);
			}

		}

		//sw.tick("Write offsets: ", 2);
		
	}

	void finalize_offsets()
	{

	}

};




template<typename T>
void write_offsets(const params& a, std::vector<vec2<T>>& offsets, const std::vector<std::array<T, 6>>& coefs)
{
	for (size_t i = 0; i < offsets.size(); ++i)
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
