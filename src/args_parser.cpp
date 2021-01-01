#include "args_parser.hpp"
#include "args_parser.hpp"

#include <array>
#include <fstream>

#include "option.h"

#include "slice_picture.hpp"
#include "stopwatch.hpp"
#include "subpixel_max.hpp"
#include "load_tiff.hpp"

namespace mbas {
template<std::size_t max>
bool get_ints(const std::string& value, std::array<uint32_t, max>& result)
{
	std::string token;
	std::istringstream tokenStream(value);
	uint32_t i = 0;
	while (i < max && std::getline(tokenStream, token, ','))
	{
		try
		{
			result[i] = std::stoul(token);
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
		std::array<uint32_t, 4> ints;
		if (!get_ints(value, ints))
			return false;
		result = { {(emida::esize_t)ints[0], (emida::esize_t)ints[1]}, {(emida::esize_t)ints[2], (emida::esize_t)ints[3]} };
		return true;
	}
};

template <>
struct value_type<emida::size2_t>
{
	static bool parse(std::string value, emida::size2_t& result)
	{
		std::array<uint32_t, 2> ints;
		if (!get_ints(value, ints))
			return false;
		result = { (emida::esize_t)ints[0], (emida::esize_t)ints[1] };
		return true;
	}
};
}

namespace emida {

std::pair<size_t, std::vector<size2_t>> load_slice_begins(const std::string& file_name)
{
	std::vector<size2_t> res;
	std::ifstream fin(file_name);

	std::string line;
	std::getline(fin, line);
	size_t size;
	try
	{
		size = std::stoull(line);
	}
	catch (const std::exception & e)
	{
		std::cerr << "Warning: could not parse the first line with size of file " << file_name << ". Error:'" << e.what() << "'. Skipping.\n";
	}
	int i = 0;
	while (std::getline(fin, line))
	{
		++i;
		size_t split = line.find(' ');
		if (split == std::string::npos || split >= line.size())
		{
			std::cerr << "Warning: could not parse line " << i << " of file " << file_name << ". Skipping.\n";
			continue;
		}
		std::string s_x = line.substr(0, split);
		std::string s_y = line.substr(split+1);

		try
		{
			size2_t begin;
			begin.x = std::stoul(s_x);
			begin.y = std::stoul(s_y);
			res.push_back(begin);
		}
		catch (const std::exception& e)
		{
			std::cerr << "Warning: could not parse line " << i << " of file " << file_name << ". Error:'" << e.what() << "'. Skipping.\n";
		}
	}

	fin.close();
	return { size, res };
}

bool params::parse(int argc, char** argv)
{
	using namespace mbas;

	command cmd;

	cmd.add_options()
		("i,initial", "Path to picture of initial undeformed picture.", value_type<std::string>(), "TIFF_FILE", false)
		("d,deformed", "Path to file with list of paths to deformed pictures.", value_type<std::string>(), "TXT_FILE", false)
		("o,outpics", "If specified, the program will write offsets into pictures and save them in specified folder.", value_type<std::string>(), "FOLDER")
		("c,crosssize", "Size of neighbourhood that is analyzed in each slice of picture. Must be odd numbers.", value_type<emida::size2_t>(), "X_SIZE,Y_SIZE")
		("s,slicesize", "Size of slices of pictures that are to be compared. The parameter specifies \"radius\" - half of slice size. Topleft corner of each slice is <slicepos>-<slicesize> and bottom right corner is <slicepos>+<slicesize>-1", value_type<emida::size2_t>(), "X_RADIUS,Y_RADIUS")
		("slicestep", "If slicepos not specified, specifies density of slices", value_type<emida::size2_t>(), "X_SIZE,Y_SIZE")
		("b,slicepos", "Path to file with positions of slice middles in each picture", value_type<std::string>(), "FILE_PATH", false)
		("a,analysis", "The application will write time measurements and statistics about processed files to standard error output.")
		("q,writecoefs", "In addition to offsets, output also coefficients of parabola fitting for each region of interest.")
		("precision", "Specifies the floating type to be used. Allowed values: double, float", value_type<std::string>(), "double|float")
		("f,fitsize", "Specifies size of neighbourhood that is used to fitting and finding subpixel maximum. Allowed values: 3, 5, 7, 9", value_type<int>(), "SIZE")
		("crosspolicy", "Specified whether to use FFT to compute cross correlation. Allowed values: brute, fft.", value_type<std::string>(), "brute|fft")
		("batchsize", "Specifies how many files are processed in one batch", value_type<int>(), "NUMBER")
		("loadworkers", "Specifies number of workers that load input patterns simultaneously", value_type<int>(), "NUMBER")
		("h,help", "Print a usage message on standard output and exit successfully.");
	
	auto parsed = cmd.parse(argc, argv);
	
	if (!parsed.parse_ok())
	{
		std::cerr << "Argument parsing error." << "\n";
		std::cerr << cmd.help() << "\n";
		return false;
	}
	if (parsed["help"])
	{
		std::cerr << cmd.help() << "\n";
		return true;
	}

	initial_file_name = parsed["initial"]->get_value<std::string>();
	deformed_list_file_name = parsed["deformed"]->get_value<std::string>();
	if (parsed["outpics"])
		out_dir = parsed["outpics"]->get_value<std::string>();

	if (parsed["slicesize"])
	{
		slice_size = parsed["slicesize"]->get_value<size2_t>();
		slice_size = slice_size * 2;
	}

	if (parsed["slicepos"])
	{
		if (parsed["slicestep"])
		{
			std::cerr << "Error: cannot use slicestep when slicepos specified\n";
			return false;
		}

		auto [size, loaded] = load_slice_begins(parsed["slicepos"]->get_value<std::string>());
		slice_mids = loaded;
		if (parsed["slicesize"])
			std::cerr << "Warning: size written in --slicepos file will be overriden by --slicesize.\n";
		else
			slice_size = { (esize_t) size * 2U, (esize_t)size * 2U };

		slice_begins = slice_mids;
		for (auto& o : slice_begins)
			o = o - (slice_size / 2);

		for (auto m : slice_mids)
		{
			size2_t begin = m - slice_size / 2;
			if (begin.x < 0 || begin.y < 0)
			{
				std::cerr << "Error: A slice out of bounds\n";
				return false;
			}
			size2_t end = m + slice_size / 2;
		}
	}
	
	if (parsed["crosssize"])
		cross_size = parsed["crosssize"]->get_value<size2_t>();
	else
		cross_size = slice_size * 2 - 1;

	if (cross_size.x > slice_size.x * 2 - 1 || cross_size.y > slice_size.y * 2 - 1)
	{
		std::cerr << "Error: cross size must be lesser or equal to slice_size * 2 - 1.\n";
		return false;
	}

	if (parsed["precision"])
	{
		const std::string& precision_s = parsed["precision"]->get_value<std::string>();
		if (precision_s == "float")
			precision = precision_type::FLOAT;
		else if (precision_s == "double")
			precision = precision_type::DOUBLE;
		else
		{
			std::cerr << "Invalid precision argument.\n";
			return false;
		}
	}

	if (parsed["fitsize"])
	{
		fitting_size = parsed["fitsize"]->get_value<int>();
		try
		{
			get_lstsq_matrix(fitting_size);
		}
		catch(const std::runtime_error&)
		{
			std::cerr << "Error: the fitsize " << fitting_size << " is not supported.\n";
			return false;
		}
	}

	if (parsed["crosspolicy"])
	{
		const std::string& crosspolicy = parsed["crosspolicy"]->get_value<std::string>();
		if (crosspolicy == "fft")
		{
			cross_pol = CROSS_POLICY_FFT;
			if (parsed["crosssize"])
				std::cerr << "Warning: --crosssize (-c) is ignored with FFT policy.\n";
		}
		else if (crosspolicy == "brute")
			cross_pol = CROSS_POLICY_BRUTE;
		else
		{
			std::cerr << "Invalid crosspolicy argument.\n";
			return false;
		}
		
	}

	if (parsed["batchsize"])
	{
		int val = parsed["batchsize"]->get_value<int>();
		if (val > 0)
			batch_size = val;
		else
		{
			std::cerr << "Error: batchsize must be greater than 0.";
			return 1;
		}
	}

	if (parsed["loadworkers"])
	{
		int val = parsed["loadworkers"]->get_value<int>();
		if (val > 0)
			load_workers = val;
		else
		{
			std::cerr << "Error: loadworkers must be greater than 0.";
			return 1;
		}
	}


	analysis = parsed["analysis"] ? true : false;
	write_coefs = parsed["writecoefs"] ? true : false;
	stopwatch::global_activate = analysis;

	if (!get_size(initial_file_name, pic_size))
		return false;

	return true;
}

}
