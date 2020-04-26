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
		switch (a.precision)
		{
		case precision_type::DOUBLE:
			emida::process_files<double>(a);
			break;
		case precision_type::FLOAT:
			emida::process_files<float>(a);
			break;
		default:
			break;
		}
		
	}
	catch (std::runtime_error & e)
	{
		std::cout << "Error: " << e.what() << "\n";
		return 1;
	}

	return 0;
}