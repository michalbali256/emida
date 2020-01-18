#include <iostream>

#include <option.h>
#include "cross_corr.hpp"
int main(int argc, char ** argv)
{
	/*using namespace mbas;
	command cmd;

	cmd.add_options()
		("f,format", "Specify output format, possibly overriding the format specified in the environment variable TIME.", value_type<std::string>(), "FORMAT")
		("p,portability", "Use the portable output format.")
		("o,output", "Do not send the results to stderr, but overwrite the specified file.", value_type<std::string>(), "FILE")
		("a,append", "(Used together with - o.) Do not overwrite but append.")
		("v,verbose", "Give very verbose output about all the program knows about.")
		("help", "Print a usage message on standard output and exit successfully.")
		("V,version", "Print version information on standard output, then exit successfully.");

	std::cout << cmd.help();

	auto parsed = cmd.parse(argc, argv);

	if (!parsed.parse_ok())
	{
		std::cout << "Argument parsing error." << "\n";
		std::cout << cmd.help() << "\n";
		return 1;
	}

	if (parsed["v"])
	{
		std::cout << "Version 0" << "\n";
		return 0;
	}


	std::string format;
	auto format_opt = parsed["f"];
	if (format_opt)
		format = format_opt->params()[0]->get_value<std::string>();
		*/

	size_t n = 5;
	std::vector<int> x;
	x.resize(n*n);
	for (size_t i = 0; i < n * n; ++i)
		x[i] = i;

	size_t res_n = (2 * n - 1);
	std::vector<int> res;
	res.resize((2 * n - 1) * (2 * n - 1));

	emida::cross_corr_serial<int, int>(x.data(), x.data(), res.data(), n, n);

	for (size_t i = 0; i < res_n; ++i)
	{
		for (size_t j = 0; j < res_n; ++j)
		{
			std::cout << res[i * res_n + j] << "\t";
		}
		std::cout << "\n";
	}

	return 0;
}
