#include <iostream>
#include <sstream>
#include <vector>

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

constexpr double PI = 3.14159265358979323846;

template<typename T>
inline std::vector<T> hanning(size_t size)
{
	std::vector<T> res;
	res.resize(size);
	for (size_t i = 0; i < size; i++)
		res[i] = 0.5 * (1 - cos(2 * PI * i / (size - 1)));
	return res;
}

template<typename T>
inline T* vector_to_host(const std::vector<T> & v)
{
	T* cu_ptr;
	CUCH(cudaMalloc(&cu_ptr, v.size() * sizeof(T)));
	CUCH(cudaMemcpy(cu_ptr, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
	return cu_ptr;
}

}