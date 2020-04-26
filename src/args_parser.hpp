#pragma once

#include <string>
#include <vector>
#include <optional>


#include "common.hpp"

namespace emida {

struct params
{
	bool parse(int argc, char** argv);

	std::string initial_file_name;
	std::string deformed_list_file_name;

	std::optional<std::string> out_dir;
	size2_t pic_size;
	size2_t cross_size;
	size2_t slice_size = { 64, 64 };
	std::vector<size2_t> slice_mids;
	bool analysis = false;
};





}