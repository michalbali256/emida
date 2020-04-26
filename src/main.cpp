#include <iostream>
#include <string>
#include <optional>

#include "process_files.hpp"
#include "args_parser.hpp"

int main(int argc, char ** argv)
{
	using namespace emida;
	
	params a;
	if (!a.parse(argc, argv))
		return 1;

	try
	{
		emida::process_files<double>(a);
	}
	catch (std::runtime_error & e)
	{
		std::cout << "Error: " << e.what() << "\n";
		return 1;
	}
	/*for (const auto & offset_list : offsets)
	{
		for(auto off : offset_list)
			std::cout << off.x << " " << off.y << "\n";
	}*/

	return 0;
}