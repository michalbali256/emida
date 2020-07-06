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

template<typename T, typename U>
inline __host__ __device__ T div_up(T a, U b)
{
	return (a + b - 1) / b;
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

	__host__ __device__ vec2<T> operator*(const dim3& rhs) const
	{
		return { x * rhs.x, y * rhs.y };
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

	template<typename U>
	__host__ __device__ vec2<T> operator%(const U& rhs) const
	{
		return { x % rhs, y % rhs };
	}
	__host__ __device__ vec2<T> operator%(const vec2<T>& rhs) const
	{
		return { x % rhs.x, y % rhs.y };
	}


	__host__ __device__ __inline__ size_t pos(size_t cols) const { return y * cols + x; }

	static __host__ __device__ __inline__ vec2<T> from_id(T id, size_t width) { return { id % width, id / width }; }
};

using size2_t = vec2<size_t>;
using int2_t = vec2<int>;

struct range
{
	size2_t begin;
	size2_t end;
};

enum cross_policy
{
	CROSS_POLICY_BRUTE,
	CROSS_POLICY_FFT
};

}
