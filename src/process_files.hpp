#pragma once
#include <filesystem>
#include <fstream>
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
		if(a.write_coefs)
			printf("%llu %llu %f %f %f %f %f %f %f %f\n", a.slice_mids[i].x, a.slice_mids[i].y, offsets[i].x, offsets[i].y,
				coefs[i][0], coefs[i][1], coefs[i][2], coefs[i][3], coefs[i][4], coefs[i][5]);
		else
			printf("%llu %llu %f %f\n", a.slice_mids[i].x, a.slice_mids[i].y, offsets[i].x, offsets[i].y);
	}
}

template<typename T>
inline void process_files(params& a)
{
	stopwatch sw(true, 3);

	size_t file_batch_size = a.slice_mids.size();

	size2_t pic_size;
	if (!get_size(a.initial_file_name, pic_size))
		return;

	auto slice_begins = a.slice_mids;
	for (auto& o : slice_begins)
		o = o - (a.slice_size / 2);
	
	std::vector<uint16_t> initial_raster(pic_size.area());

	if (!load_tiff(a.initial_file_name, initial_raster.data(), pic_size))
		return;
	
	gpu_offset<T, uint16_t> offs(pic_size, &slice_begins, a.slice_size, a.cross_size, a.batch_size, a.cross_pol, a.fitting_size);

	offs.allocate_memory(initial_raster.data());

	//TODO: allocate cuda host memory to avoid copying the data twice
	std::vector<uint16_t> deformed_raster(pic_size.area() * a.batch_size);

	std::vector<file> work = load_work(a.deformed_list_file_name);
	
	for (size_t c = 0; c < work.size(); c += a.batch_size)
	{
		size_t batch_files = c + a.batch_size > work.size() ? work.size() - c : a.batch_size;
		bool OK = true;
		uint16_t * next = deformed_raster.data();
		for (size_t i = c; i < c + batch_files; ++i)
		{
			OK &= load_tiff(work[i].fname, next, pic_size);
			next += pic_size.area();
		}sw.tick("Load tiff: ", 2);

		if (!OK)
			continue;


		auto [offsets, coefs] = offs.get_offset(deformed_raster.data());
		sw.tick("Get offset: ", 2);

		if (a.analysis)
			stopwatch::global_stats.inc_histogram(offsets);
		for (size_t j = c ; j < c + batch_files; ++j)
		{
			printf("%f %f %llu %s\n", work[j].x, work[j].y, file_batch_size, work[j].fname.c_str());
			size_t it = j - c;
			for (size_t i = it * file_batch_size; i < (it+1) * file_batch_size; ++i)
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
		
		sw.tick("Write offsets: ", 2);

		sw.tick("ONE: ", 1);
	}

	sw.total();
	if (a.analysis)
	{
		std::cerr << "Total pics: " << stopwatch::global_stats.total_pics << "\n";
		stopwatch::global_stats.write_histogram();
	}
}

}
