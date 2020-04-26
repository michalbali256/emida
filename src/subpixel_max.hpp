#pragma once

#include <array>

#include "common.hpp"

namespace emida
{

template<int s>
constexpr auto get_matrix()
{
	std::array<double, s * s * 6> ret;

	size_t i = 0;
	for (size_t y = 0; y < s; ++y)
	{
		for (size_t x = 0; x < s; ++x)
		{
			ret[i * 6] = 1;
			ret[i * 6 + 1] = (double)x;
			ret[i * 6 + 2] = (double)y;
			ret[i * 6 + 3] = (double)x * x;
			ret[i * 6 + 4] = (double)x * y;
			ret[i * 6 + 5] = (double)y * y;
			++i;
		}
	}

	return ret;
}

//a = m x k
//b = k x n
//c = m x n
template <typename F1, typename F2, typename F3>
void multiply(const F1* a, const F2* b, F3* c, size_t m, size_t n, size_t k)
{
	for (size_t x = 0; x < n; ++x)
	{
		for (size_t y = 0; y < m; ++y)
		{
			F3 sum = 0;
			for (size_t i = 0; i < k; ++i)
			{
				sum += a[y * k + i] * b[i * n + x];
			}
			c[y * n + x] = sum;
		}
	}
}

/*
template <>
inline void multiply<double, half, half>(const double* a, const half* b, half* c, size_t m, size_t n, size_t k)
{
	for (size_t x = 0; x < n; ++x)
	{
		for (size_t y = 0; y < m; ++y)
		{
			half sum = 0.0;
			for (size_t i = 0; i < k; ++i)
			{
				sum = sum + (half)(a[y * k + i] * b[i * n + x]);
			}
			c[y * n + x] = sum;
		}
	}
}*/

template<typename T, int s>
struct lstsq_matrix
{
	static const std::array<T, 6 * s * s> mat;
};

//coefficients of quadratic function of neighborhood of maximum
//f(x,y) = a + bx + cy + dx^2 + exy + fy^2
template<typename T, int s>
std::array<T, 6> subpixel_max_coefs(const T* pic)
{
	//we get the values of s*s points, a-f are the unknown variables
	//f(x,y) = a + xb + yc + x^2 * d + xye + y^2 * f
	// for s = 3, A =
	//[[1. 0. 0. 0. 0. 0.
	// [1. 1. 0. 1. 0. 0.]
	// [1. 2. 0. 4. 0. 0.]
	// [1. 0. 1. 0. 0. 1.]
	// [1. 1. 1. 1. 1. 1.]
	// [1. 2. 1. 4. 2. 1.]
	// [1. 0. 2. 0. 0. 4.]
	// [1. 1. 2. 1. 2. 4.]
	// [1. 2. 2. 4. 4. 4.]]

	//https://en.wikipedia.org/wiki/Least_squares#Linear_least_squares
	//lstsq_matrix<T, s>::mat is (A^T * A)^-1 * A^T
	//it is precomputed since we know it at compile time
	std::array<T, 6> coef;
	multiply(lstsq_matrix<double, s>::mat.data(), pic, coef.data(), 6, 1, s * s);
	return coef;
}

template<typename T, int s>
std::vector<vec2<T>> subpixel_max_serial(const T* pic, size_t batch_size)
{
	std::vector<vec2<T>> ret(batch_size);
	for (size_t i = 0; i < batch_size; ++i, pic += s*s)
	{
		//coefficients of quadratic function of neighborhood of maximum
		//f(x,y) = a + bx + cy + dx^2 + exy + fy^2
		auto [a, b, c, d, e, f] = subpixel_max_coefs<T, s>(pic);

		//now get the maximum of that function. Partial derivations:
		//from partial derivation by x : 2dx + ey + b = 0
		//from partial derivation by y : ex + 2fy + c = 0
		//solve the 2 equations:
		ret[i].y = (b * e - 2 * c * d) / (4 * f * d - e * e);
		ret[i].x = (-e * ret[i].y - b) / (2 * d);
	}
	return ret;
}


}
