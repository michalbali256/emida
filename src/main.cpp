#include <iostream>
#include <string>
#include <optional>

#include "process_files.hpp"
#include "args_parser.hpp"
#include "stopwatch.hpp"

int main(int argc, char ** argv)
{
	using namespace emida;
	
	params a;
	if (!a.parse(argc, argv))
		return 1;

	try
	{
		if (a.precision == precision_type::DOUBLE)
		{
			emida::file_processor<double> p(a);
			p.process_files();
		}
		else if ((a.precision == precision_type::FLOAT))
		{
			emida::file_processor<float> p(a);
			p.process_files();
		}
	}
	catch (std::runtime_error & e)
	{
		std::cout << "Error: " << e.what() << "\n";
		return 1;
	}

	stopwatch::write_durations();
	
	return 0;
}