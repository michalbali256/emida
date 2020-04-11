#include <string>
#include <vector>
#include <optional>


#include "common.hpp"

namespace emida {

struct args_parser
{
	bool parse(int argc, char** argv);



	std::string initial_dir;
	std::string deformed_dir;
	range files_range;
	std::optional<std::string> out_dir;
	size2_t pic_size;
	size2_t cross_size;
	size2_t slice_size = { 64, 64 };
	std::vector<size2_t> slice_begins;
};





}