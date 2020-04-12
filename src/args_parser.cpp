#include "args_parser.hpp"

#include <array>
#include <fstream>

#include "option.h"

#include "slice_picture.hpp"

namespace mbas {
template<std::size_t max>
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
		if (!get_ints(value, ints))
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

namespace emida {

std::vector<size2_t> load_slice_begins(const std::string & file_name)
{
	std::vector<size2_t> res;
	std::fstream fin(file_name);
	std::string line;
	int i = 0;
	while (std::getline(fin, line))
	{
		size2_t begin;
		
		if (mbas::value_type<emida::size2_t>::parse(std::move(line), begin))
			res.push_back(begin);
		else
			std::cerr << "Warning: could not parse line " << i << " of file " << file_name << ". Skipping.\n";
	}

	fin.close();
	return res;
}

bool params::parse(int argc, char** argv)
{
	using namespace mbas;

	command cmd;

	cmd.add_options()
		("i,initial", "Path to folder with initial undeformed pictures.", value_type<std::string>(), "FOLDER", false)
		("d,deformed", "Path to folder with deformed pictures.", value_type<std::string>(), "FOLDER", false)
		("o,outpics", "If specified, the program will write offsets into pictures and save them in specified folder.", value_type<std::string>(), "FOLDER")
		("r,range", "Defines a range of pictures to be compared", value_type<range>(), "X_BEGIN,Y_BEGIN,X_END,Y_END", false)
		("c,crosssize", "Size of neighbourhood that is analyzed in each slice of picture. Must be odd numbers.", value_type<emida::size2_t>(), "X_SIZE,Y_SIZE")
		("p,picsize", "Size of one input tiff picture. Default is 873,873", value_type<emida::size2_t>(), "X_SIZE,Y_SIZE")
		("s,slicesize", "Size of slices of pictures that are to be compared", value_type<emida::size2_t>(), "X_SIZE,Y_SIZE")
		("slicestep", "If slicepos not specified, specifies density of slices", value_type<emida::size2_t>(), "X_SIZE,Y_SIZE")
		("b,slicepos", "Path to file with positions of slices in each picture", value_type<std::string>(), "FILE_PATH")
		//("a,analysis", "Output analysis results")
		("h,help", "Print a usage message on standard output and exit successfully.");

	auto parsed = cmd.parse(argc, argv);

	if (!parsed.parse_ok())
	{
		std::cout << "Argument parsing error." << "\n";
		std::cout << cmd.help() << "\n";
		return false;
	}
	if (parsed["help"])
	{
		std::cout << cmd.help() << "\n";
		return true;
	}

	initial_dir = parsed["initial"]->get_value<std::string>();
	deformed_dir = parsed["deformed"]->get_value<std::string>();
	if (parsed["outpics"])
		out_dir = parsed["outpics"]->get_value<std::string>();
	files_range = parsed["range"]->get_value<range>();

	
	if (parsed["picsize"])
		pic_size = parsed["picsize"]->get_value<size2_t>();
	else
		pic_size = { 873, 873 };

	
	if (parsed["crosssize"])
		cross_size = parsed["crosssize"]->get_value<size2_t>();
	else
		cross_size = pic_size * 2 - 1;

	if (parsed["slicesize"])
		slice_size = parsed["slicesize"]->get_value<size2_t>();

	if (parsed["slicepos"])
	{
		if (parsed["slicestep"])
		{
			std::cerr << "Error: cannot use slicestep when slicepos specified\n";
			return false;
		}
		slice_begins = load_slice_begins(parsed["slicepos"]->get_value<std::string>());
	}
	else
	{
		size2_t step = { 32, 32 };
		if (parsed["slicestep"])
			step = parsed["slicestep"]->get_value<size2_t>();
		slice_begins = get_slice_begins(pic_size, slice_size, step);
	}

	return true;
}



}
