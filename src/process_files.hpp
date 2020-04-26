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

inline std::string append_filename(const std::string& dir, const std::string& file_name)
{
	return (std::filesystem::path(dir) / file_name).string();
}

template<typename T>
inline std::vector<std::vector<vec2<T>>> process_files(const params& a)
{
	stopwatch sw(true, 3);

	auto slice_mids = a.slice_begins;
	for (auto& o : slice_mids)
		o = o + (a.slice_size / 2);
	

	std::vector<std::vector<vec2<T>>> res;


	gpu_offset<T, uint16_t> offs(a.pic_size, &a.slice_begins, a.slice_size, a.cross_size);
	offs.allocate_memory();

	//TODO: allocate cuda host memory to avoid copying the data twice
	std::vector<uint16_t> initial_raster(a.pic_size.area()); 
	if (!load_tiff(a.initial_dir, initial_raster.data(), a.pic_size))
		return res;

	std::ifstream infile(a.deformed_dir);
	std::string line;
	while(std::getline(infile,line))
	{
		std::stringstream iss(line);
		double x,y;
		std::string fname;
		iss >> x;
		iss >> y;
		iss.ignore();
		std::getline(iss, fname);
		printf("%f %f %ld %s\n", x, y, a.slice_begins.size(), fname.c_str());


		//TODO: allocate cuda host memory to avoid copying the data twice
		std::vector<uint16_t> deformed_raster(a.pic_size.area());
		bool OK = true;
		OK &= load_tiff(fname, deformed_raster.data(), a.pic_size); sw.tick("Load tiff: ", 2);
		if (!OK)
			continue;

		auto offsets = offs.get_offset(deformed_raster.data(), initial_raster.data());
		sw.tick("Get offset: ", 2);

		if (a.analysis)
			stopwatch::global_stats.inc_histogram(offsets);


		for (size_t i = 0; i < offsets.size(); ++i)
		{
			//sometimes the resulting float is outputted as nan and sometimes as nan(ind). Normalize that here.
			if (isnan(offsets[i].x))
				offsets[i].x = std::numeric_limits<T>::quiet_NaN();
			if (isnan(offsets[i].y))
				offsets[i].y = std::numeric_limits<T>::quiet_NaN();
			printf("%lu %lu %f %f\n", slice_mids[i].x, slice_mids[i].y, offsets[i].x, offsets[i].y);
		}
		sw.tick("Write offsets: ", 2);

		res.push_back(std::move(offsets));

		sw.tick("ONE: ", 1);

	}

	sw.total();
	if (a.analysis)
	{
		std::cerr << "Border X: " << stopwatch::global_stats.border.x << "\n";
		std::cerr << "Border Y: " << stopwatch::global_stats.border.y << "\n";
		std::cerr << "Total pics: " << stopwatch::global_stats.total_pics << "\n";
		stopwatch::global_stats.write_histogram();
	}
	return res;
}

}
