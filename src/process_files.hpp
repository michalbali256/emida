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
	std::string initial_prefix = append_filename(a.initial_dir, "INITIAL_");
	std::string deformed_prefix = append_filename(a.deformed_dir, "DEFORMED_");
	std::string out_prefix;
	if(a.out_dir)
		out_prefix = append_filename(*a.out_dir, "OUT_");

	gpu_offset<double> offs(a.slice_size, a.cross_size, a.slice_begins.size());
	offs.allocate_memory();

	std::cout << a.slice_begins.size() << "\n";

	for (size_t j = a.files_range.begin.y; j < a.files_range.end.y; ++j)
	{
		for (size_t i = a.files_range.begin.x; i < a.files_range.end.x; ++i)
		{
			size_t x = i * 60 + (j % 2 * 30);
			size_t y = j * sqrt(0.75) * 60;
			std::string file_suffix = "x" + std::to_string(x) + "y" + std::to_string(y) + ".tif";

			std::vector<uint16_t> initial_raster(a.pic_size.area());
			std::vector<uint16_t> deformed_raster(a.pic_size.area());
			load_tiff(initial_prefix + file_suffix, initial_raster.data(), a.pic_size);
			load_tiff(deformed_prefix + file_suffix, deformed_raster.data(), a.pic_size); sw.tick("Load tiff: ", 2);


			auto initial_slices = get_pics<double>(initial_raster.data(), a.pic_size, a.slice_begins, a.slice_size);
			auto deformed_slices = get_pics<double>(deformed_raster.data(), a.pic_size, a.slice_begins, a.slice_size);
			sw.tick("Create slices: ", 2);

			auto offsets = offs.get_offset(initial_slices.data(), deformed_slices.data());
			sw.tick("Get offset: ", 2);

			if (a.out_dir)
				draw_tiff(deformed_raster.data(), a.pic_size, out_prefix + file_suffix, offsets, slice_mids);
			sw.tick("Draw tiff: ", 2);

			
			

			std::cout << "x" << x << "y" << y << "\n";
			for(size_t i = 0; i < offsets.size(); ++i)
				std::cout << a.slice_begins[i].x << " " << a.slice_begins[i].y << " " << offsets[i].x << " " << offsets[i].y << "\n";
			sw.tick("Write offsets: ", 2);

			res.push_back(std::move(offsets));

			sw.tick("ONE: ", 1);

		}

	}
	sw.total();
	std::cerr << "Border X: " << stopwatch::stats.border.x << "\n";
	std::cerr << "Border Y: " << stopwatch::stats.border.y << "\n";
	std::cerr << "Total pics: " << stopwatch::stats.total_pics << "\n";
	return res;
}

}