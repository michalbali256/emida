#include <iostream>

#include <option.h>
#include "cross_corr.hpp"
#include "matrix.hpp"
#include <chrono>
#include "cross_corr_host.hpp"

#include <filesystem>


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

	if (parsed.plain_args().size() != 1)
	{
		std::cout << "The program processes exactly one input.\n";
		return 1;
	}
	file_name = parsed.plain_args()[0];
	using namespace emida;
	matrix<int> a, b, res_expected;
	try
	{
		a = matrix<int>::from_file(file_name + "_A.txt");
		b = matrix<int>::from_file(file_name + "_B.txt");
		res_expected = matrix<int>::from_file(file_name + "_res.txt");
	}
	catch (const std::exception & e)
	{
		std::cout << e.what() << "\n";
	}

	if (a.n != b.n)
	{
		std::cout << "Matrices A and B have different sizes, exiting.\n";
		return 1;
	}


	size_t res_n = (2 * a.n - 1);
	if (res_expected.n != res_n)
	{
		std::cout << "Res matrix has incompatible size with a and b.\n";
		return 1;
	}
	std::vector<int> res;

	/*
	algorithm_cross_corr acc;
	std::chrono::high_resolution_clock c;
	auto start = c.now();


	cudaSetDevice(0);
	
	
	acc.prepare(b, a);
	write_duration("Preparation: ", start, c.now());
	start = c.now();
	acc.run();
	write_duration("Run: ", start, c.now());
	start = c.now();
	acc.finalize();
	write_duration("Finalization: ", start, c.now());

	if (acc.result().data != res_expected.data)
	{
		std::cout << "Result is wrong\n";
	}
	else
	{
		std::cout << "Result is OK\n";
	}
	*/
	return 0;
}