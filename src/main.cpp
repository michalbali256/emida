#include <iostream>
#include <string>
#include <optional>

#include "process_files.hpp"
#include "args_parser.hpp"

int main(int argc, char ** argv)
{
	using namespace emida;
	
	args_parser p;
	if (!p.parse(argc, argv))
		return 1;

	auto offsets = emida::process_files(
		p.initial_dir,
		p.deformed_dir,
		p.out_dir ? &*p.out_dir : nullptr,
		p.files_range,
		p.pic_size,
		p.cross_size,
		p.slice_size,
		p.slice_begins);

	/*for (const auto & offset_list : offsets)
	{
		for(auto off : offset_list)
			std::cout << off.x << " " << off.y << "\n";
	}*/

	return 0;
}