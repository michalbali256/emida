#pragma once
#include <filesystem>

#include "get_offset.hpp"
#include "load_tiff.hpp"
#include "slice_picture.hpp"
#include "draw_tiff.hpp"

namespace emida
{

inline std::string append_filename(const std::string& dir, const std::string& file_name)
{
	return (std::filesystem::path(dir) / file_name).string();
}

inline std::vector<std::vector<vec2<double>>> process_files(
	const std::string& initial_dir,
	const std::string& deformed_dir,
	const std::string* out_dir,
	range files_range,
	size2_t one_size,
	size2_t cross_size)
{
	stopwatch sw(true, 3);
	vec2<size_t> slice_size = { 64, 64 };
	auto slice_begins = get_slice_begins(one_size, slice_size, { 32, 32 });

	auto slice_mids = slice_begins;
	for (auto& o : slice_mids)
		o = o + (slice_size / 2);
	

	std::vector<std::vector<vec2<double>>> res;
	std::string initial_prefix = append_filename(initial_dir, "INITIAL_");
	std::string deformed_prefix = append_filename(deformed_dir, "DEFORMED_");
	std::string out_prefix;
	if(out_dir)
		out_prefix = append_filename(*out_dir, "OUT_");

	gpu_offset<double> offs(slice_size, cross_size, slice_begins.size());
	offs.allocate_memory();

	for (size_t j = files_range.begin.y; j < files_range.end.y; ++j)
	{
		for (size_t i = files_range.begin.x; i < files_range.end.x; ++i)
		{
			size_t x = i * 60 + (j % 2 * 30);
			size_t y = j * sqrt(0.75) * 60;
			std::string file_suffix = "x" + std::to_string(x) + "y" + std::to_string(y) + ".tif";

			std::vector<uint16_t> initial_raster(one_size.area());
			std::vector<uint16_t> deformed_raster(one_size.area());
			load_tiff(initial_prefix + file_suffix, initial_raster.data(), one_size);
			load_tiff(deformed_prefix + file_suffix, deformed_raster.data(), one_size); sw.tick("Load tiff: ", 2);


			auto initial_slices = get_pics<double>(initial_raster.data(), one_size, slice_begins, slice_size);
			auto deformed_slices = get_pics<double>(deformed_raster.data(), one_size, slice_begins, slice_size);
			sw.tick("Create slices: ", 2);

			auto offsets = offs.get_offset(initial_slices.data(), deformed_slices.data());
			sw.tick("Get offset: ", 2);

			if (out_dir)
				draw_tiff(deformed_raster.data(), one_size, out_prefix + file_suffix, offsets, slice_mids);
			sw.tick("Draw tiff: ", 2);

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