#pragma once

#include <iostream>
#include <sstream>
#include <vector>
#include <type_traits>
#include <stdexcept>
#include <cmath>

#include "cuda_runtime.h"

namespace emida
{

#define CUCH(status) emida::cuda_check(status, __LINE__, __FILE__, #status)

template<typename T, typename U>
inline __host__ __device__ T div_up(T a, U b)
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
inline T* vector_to_device(const std::vector<T> & v)
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
void copy_to_device(T * data, size_t size, T* cu_ptr)
{
	CUCH(cudaMemcpy(cu_ptr, data, size * sizeof(T), cudaMemcpyHostToDevice));
}

template<typename T>
T * cuda_malloc(size_t num_elements)
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

template <typename T>
struct data_index
{
	T data;
	size_t index;
};

template<typename T>
struct vec2
{
	T x;
	T y;
	__host__ __device__ T area() const { return x * y; }

	__host__ __device__ vec2<T> operator+(const vec2<T>& rhs) const
	{
		return { x + rhs.x, y + rhs.y };
	}
	template<typename U>
	__host__ __device__ vec2<T> operator+(const U& rhs) const
	{
		return { x + rhs, y + rhs };
	}
	template<typename U>
	friend __host__ __device__ vec2<T> operator+(const U& lhs, const vec2<T>& rhs)
	{
		return { lhs + rhs.x, lhs + rhs.y };
	}

	__host__ __device__ vec2<T> operator-() const
	{
		return { -x, -y };
	}
	__host__ __device__ vec2<T> operator-(const vec2<T>& rhs) const
	{
		return { x - rhs.x, y - rhs.y };
	}
	template<typename U>
	__host__ __device__ vec2<T> operator-(const U& rhs) const
	{
		return { x - rhs, y - rhs };
	}

	__host__ __device__ vec2<T> operator*(const vec2<T>& rhs) const
	{
		return { x * rhs.x, y * rhs.y };
	}
	
	template<typename U>
	friend __host__ __device__ vec2<T> operator*(const U& lhs, const vec2<T>& rhs)
	{
		return { lhs * rhs.x, lhs * rhs.y };
	}
	template<typename U>
	__host__ __device__ vec2<T> operator*(const U& rhs) const
	{
		return { x * rhs, y * rhs };
	}

	
	__host__ __device__ vec2<T> operator/(const vec2<T>& rhs) const
	{
		return { x / rhs.x, y / rhs.y };
	}
	template<typename U>
	__host__ __device__ vec2<T> operator/(const U& rhs) const
	{
		return { x / rhs, y / rhs };
	}

	__host__ __device__ size_t pos(size_t cols) const { return y * cols + x; }
};

using size2_t = vec2<size_t>;

struct range
{
	size2_t begin;
	size2_t end;
};

}
