#include <iostream>

#include <option.h>

#include "process_files.hpp"

int main(int argc, char ** argv)
{
	using namespace mbas;
	command cmd;

	cmd.add_options()
		("a,algorithm", "Choose algorithm which to use", value_type<std::string>(), "ALGORITHM")
		("h,help", "Print a usage message on standard output and exit successfully.");

	auto parsed = cmd.parse(argc, argv);

	if (!parsed.parse_ok())
	{
		std::cout << "Argument parsing error." << "\n";
		std::cout << cmd.help() << "\n";
		return 1;
	}
	if (parsed["help"])
	{
		std::cout << cmd.help() << "\n";
		return 0;
	}

	std::string algorithm;
	if (parsed["a"])
		algorithm = parsed["a"]->params()[0]->get_value<std::string>();
	std::string file_name;

	auto offsets = emida::process_files("../../data/FeAl/INITIAL_FeAl", "../../data/FeAl/DEFORMED_FeAl", "../../data/FeAl/OUT_FeAl", { 10, 10 });

		/*for (const auto & offset_list : offsets)
	{
		for(auto off : offset_list)
			std::cout << off.x << " " << off.y << "\n";
	}*/

	return 0;
}