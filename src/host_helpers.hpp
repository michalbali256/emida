#pragma once

#include "cuda_runtime.h"
#include <stdexcept>
#include <sstream>

namespace emida
{

#define CUCH(status) emida::cuda_check(status, __LINE__, __FILE__, #status)

inline void cuda_check(cudaError_t status, int line, const char* src_filename, const char* line_str = nullptr)
{
	if (status != cudaSuccess)
	{
		std::stringstream ss;
		ss << "CUDA Error " << status << ":" << cudaGetErrorString(status) << " in " << src_filename << " (" << line << "):" << line_str << "\n";
		std::cout << ss.str();
		throw std::runtime_error(ss.str());
	}
}

constexpr double PI = 3.14159265358979323846;

template<typename T>
inline std::vector<T> generate_hanning(size_t size)
{
	std::vector<T> res;
	res.resize(size);
	for (size_t i = 0; i < size; i++)
		res[i] = 0.5 * (1 - cos(2 * PI * i / (size - 1)));
	return res;
}

template<typename T>
inline T* vector_to_device(const std::vector<T>& v)
{
	T* cu_ptr;
	CUCH(cudaMalloc(&cu_ptr, v.size() * sizeof(T)));
	CUCH(cudaMemcpy(cu_ptr, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
	return cu_ptr;
}

template<typename T>
void copy_to_device(const std::vector<T>& v, T* cu_ptr)
{
	CUCH(cudaMemcpy(cu_ptr, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
}

template<typename T>
void copy_to_device(T* data, size_t size, T* cu_ptr)
{
	CUCH(cudaMemcpy(cu_ptr, data, size * sizeof(T), cudaMemcpyHostToDevice));
}

template<typename T>
T* cuda_malloc(size_t num_elements)
{
	T* cu_ptr;
	CUCH(cudaMalloc(&cu_ptr, num_elements * sizeof(T)));
	return cu_ptr;
}

template<typename T>
inline T* vector_to_device(const T* data, size_t size)
{
	T* cu_ptr;
	CUCH(cudaMalloc(&cu_ptr, size * sizeof(T)));
	CUCH(cudaMemcpy(cu_ptr, data, size * sizeof(T), cudaMemcpyHostToDevice));
	return cu_ptr;
}

template<typename T>
inline std::vector<T> device_to_vector(const T* cu_data, size_t size)
{
	std::vector<T> res(size);
	CUCH(cudaMemcpy(res.data(), cu_data, size * sizeof(T), cudaMemcpyDeviceToHost));
	return res;
}

template<typename T>
inline std::vector<T> repeat_vector(const std::vector<T>& vec, size_t repeat)
{
	std::vector<T> res(vec.size() * repeat);

	T* dst = res.data();
	for (size_t i = 0; i < repeat; ++i)
	{
		memcpy(dst, vec.data(), vec.size() * sizeof(T));

		dst += vec.size();
	}

	return res;
}



}