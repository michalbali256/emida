#include <iostream>
#include <string>
#include <optional>

#include <option.h>

#include "process_files.hpp"

namespace mbas {
template<int max>
bool get_ints(const std::string& value, std::array<size_t, max>& result)
{
	std::string token;
	std::istringstream tokenStream(value);
	size_t i = 0;
	while (i < max && std::getline(tokenStream, token, ','))
	{
		try
		{
			result[i] = std::stoull(token);
		}
		catch (...)
		{
			return false;
		}
		++i;
	}
	if (i != max)
		return false;
	return true;
}

template <>
struct value_type<emida::range>
{
	static bool parse(std::string value, emida::range& result)
	{
		std::array<size_t, 4> ints;
		if(!get_ints(value, ints))
			return false;
		result = { {ints[0], ints[1]}, {ints[2], ints[3]} };
		return true;
	}
};

template <>
struct value_type<emida::size2_t>
{
	static bool parse(std::string value, emida::size2_t& result)
	{
		std::array<size_t, 2> ints;
		if (!get_ints(value, ints))
			return false;
		result = { ints[0], ints[1] };
		return true;
	}
};
}


int main(int argc, char ** argv)
{
	using namespace mbas;
	using namespace emida;
	command cmd;

	cmd.add_options()
		("i,initial", "Path to folder with initial undeformed pictures.", value_type<std::string>(), "FOLDER", false)
		("d,deformed", "Path to folder with deformed pictures.", value_type<std::string>(), "FOLDER", false)
		("o,outpics", "If specified, the program will write offsets into pictures and save them in specified folder.", value_type<std::string>(), "FOLDER")
		("r,range", "Defines a range of pictures to be compared", value_type<range>(), "X_BEGIN,Y_BEGIN,X_END,Y_END", false)
		("c,crosssize", "Size of neighbourhood that is analyzed in each slice of picture. Must be odd numbers.", value_type<emida::size2_t>(), "X_SIZE,Y_SIZE")
		("s,picsize", "Size of one picture. Default is 873,873", value_type<emida::size2_t>(), "X_SIZE,Y_SIZE")
		("a,analysis", "Output analysis results")
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

	const std::string& initial_dir = parsed["initial"]->get_value<std::string>();
	const std::string& deformed_dir = parsed["deformed"]->get_value<std::string>();
	const std::string* out_dir = nullptr;
	if(parsed["outpics"])
		out_dir = &parsed["deformed"]->get_value<std::string>();
	range files_range = parsed["range"]->get_value<range>();

	size2_t pic_size;
	if (parsed["picsize"])
		pic_size = parsed["picsize"]->get_value<size2_t>();
	else
		pic_size = { 873, 873 };

	size2_t cross_size;
	if (parsed["crosssize"])
		cross_size = parsed["crosssize"]->get_value<size2_t>();
	else
		cross_size = pic_size * 2 - 1;


	

	//auto offsets = emida::process_files("../../data/FeAl/INITIAL_FeAl", "../../data/FeAl/DEFORMED_FeAl", "../../data/FeAl/OUT_FeAl", { 10, 10 });
	auto offsets = emida::process_files(
		initial_dir,
		deformed_dir,
		out_dir,
		files_range,
		pic_size,
		cross_size);

		/*for (const auto & offset_list : offsets)
	{
		for(auto off : offset_list)
			std::cout << off.x << " " << off.y << "\n";
	}*/

	return 0;
}