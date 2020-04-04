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

inline std::vector<std::vector<vec2<double>>> process_files(const std::string & initial_dir, const std::string& deformed_dir, vec2<size_t> size)
{
	vec2<size_t> one_size = { 873, 873 };
	vec2<size_t> slice_size = { 64, 64 };
	auto slice_begins = get_slice_begins(one_size, slice_size, { 32, 32 });

	std::vector<std::vector<vec2<double>>> res;
	std::string initial_prefix = append_filename(initial_dir, "INITIAL_");
	std::string deformed_prefix = append_filename(deformed_dir, "DEFORMED_");
	for (size_t j = 0; j < size.y; ++j)
	{
		for (size_t i = 0; i < size.x; ++i)
		{
			size_t x = i * 60 + (j % 2 * 30);
			size_t y = j * sqrt(0.75) * 60;
			std::string file_suffix = "x" + std::to_string(x) + "y" + std::to_string(y) + ".tif";
			
			std::vector<uint16_t> initial_raster(one_size.area());
			std::vector<uint16_t> deformed_raster(one_size.area());
			load_tiff(initial_prefix + file_suffix, initial_raster.data(), one_size);
			load_tiff(deformed_prefix + file_suffix, deformed_raster.data(), one_size);

			
			auto initial_slices = get_pics<double>(initial_raster.data(), one_size, slice_begins, slice_size);
			auto deformed_slices = get_pics<double>(deformed_raster.data(), one_size, slice_begins, slice_size);

			auto offsets = get_offset(initial_slices.data(), deformed_slices.data(), slice_size, slice_begins.size());
			
			

			draw_tiff(deformed_raster.data(), one_size, "../../vectors.tif", offsets, slice_begins, (uint16_t) 0);

			res.push_back(std::move(offsets));

		}

	}
	return res;
}

}