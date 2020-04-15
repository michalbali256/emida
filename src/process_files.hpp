#pragma once
#include <filesystem>

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

inline std::vector<std::vector<vec2<double>>> process_files(const params& a)
{
	stopwatch sw(true, 3);

	auto slice_mids = a.slice_begins;
	for (auto& o : slice_mids)
		o = o + (a.slice_size / 2);
	

	std::vector<std::vector<vec2<double>>> res;
	std::string initial_prefix = append_filename(a.initial_dir, a.initial_prefix);
	std::string deformed_prefix = append_filename(a.deformed_dir, a.deformed_prefix);
	std::string out_prefix;
	if(a.out_dir)
		out_prefix = append_filename(*a.out_dir, "OUT_");

	gpu_offset<double, uint16_t> offs(a.pic_size, &a.slice_begins, a.slice_size, a.cross_size);
	offs.allocate_memory();

	std::cout << a.slice_begins.size() << "\n";

	for (size_t j = a.files_range.begin.y; j < a.files_range.end.y; ++j)
	{
		for (size_t i = a.files_range.begin.x; i < a.files_range.end.x; ++i)
		{
			size_t x = i * 60 + (j % 2 * 30);
			size_t y =(size_t)(j * sqrt(0.75) * 60);
			std::string file_suffix = "x" + std::to_string(x) + "y" + std::to_string(y) + ".tif";

			//TODO: allocate cuda host memory to avoid copying the data twice
			std::vector<uint16_t> initial_raster(a.pic_size.area()); 
			std::vector<uint16_t> deformed_raster(a.pic_size.area());
			bool OK = true;
			OK &= load_tiff(initial_prefix + file_suffix, initial_raster.data(), a.pic_size);
			OK &= load_tiff(deformed_prefix + file_suffix, deformed_raster.data(), a.pic_size); sw.tick("Load tiff: ", 2);
			if (!OK)
				continue;

			auto offsets = offs.get_offset(deformed_raster.data(), initial_raster.data());
			sw.tick("Get offset: ", 2);

			if (a.out_dir)
				draw_tiff(deformed_raster.data(), a.pic_size, out_prefix + file_suffix, offsets, slice_mids);
			sw.tick("Draw tiff: ", 2);

			if (a.analysis)
				stopwatch::global_stats.inc_histogram(offsets);


			std::cout << "x" << x << "y" << y << "\n";
			for(size_t i = 0; i < offsets.size(); ++i)
				std::cout << a.slice_begins[i].x << " " << a.slice_begins[i].y << " " << offsets[i].x << " " << offsets[i].y << "\n";
			sw.tick("Write offsets: ", 2);

			res.push_back(std::move(offsets));

			sw.tick("ONE: ", 1);

		}

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