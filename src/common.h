#include <iostream>
#include <sstream>

namespace emida
{

#define CUCH(status) emida::cuda_check(status, __LINE__, __FILE__, #status)

template<typename T, typename U>
inline T div_up(T a, U b)
{
	return (a + b - 1) / b;
}

inline void cuda_check(cudaError_t status, int line, const char* src_filename, const char* line_str = nullptr)
{
	if (status != cudaSuccess)
	{
		std::stringstream ss;
		ss << "CUDA Error " << status << ":" << cudaGetErrorString(status) << " in " << src_filename << " (" << line << "):" << line_str << "\n";
		std::cout << ss.str();
		throw std::exception(ss.str().c_str());
	}
}

}