#pragma once

#include <array>

#include "common.hpp"

namespace emida
{

//Multiplies the a and b matrices and stores the result in c. The size of matrices are as follows:
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
				sum += (F3)(a[y * k + i] * b[i * n + x]);
			}
			c[y * n + x] = sum;
		}
	}
}

// Contains the precomputed matrix used to compute least squares (A^T * A)^-1 * A^T
template<typename T, int s>
struct lstsq_matrix
{
	static const std::array<T, 6 * s * s> mat;
	
};
const double* get_lstsq_matrix(int s);

//coefficients of quadratic function of neighborhood of maximum
//f(x,y) = a + bx + cy + dx^2 + exy + fy^2
template<typename T>
std::array<double, 6> subpixel_max_coefs(const T* pic, int s)
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
	std::array<double, 6> coef;
	multiply(get_lstsq_matrix(s), pic, coef.data(), 6, 1, s * s);
	return coef;
}

template<typename T>
struct offsets_t
{
	std::vector<vec2<T>> offsets;
	std::vector<std::array<T, 6>> coefs;
};

template<typename T>
offsets_t<double> subpixel_max_serial(const T* pic, int s, size_t batch_size)
{
	std::vector<vec2<double>> offsets(batch_size);
	std::vector<std::array<double, 6>> coefs(batch_size);
	for (size_t i = 0; i < batch_size; ++i, pic += s*s)
	{
		//coefficients of quadratic function of neighborhood of maximum
		//f(x,y) = a + bx + cy + dx^2 + exy + fy^2
		auto [a, b, c, d, e, f] = subpixel_max_coefs<T>(pic, s);

		//now get the maximum of that function. Partial derivations:
		//from partial derivation by x : 2dx + ey + b = 0
		//from partial derivation by y : ex + 2fy + c = 0
		//solve the 2 equations:
		offsets[i].y = (b * e - 2 * c * d) / (4 * f * d - e * e);
		offsets[i].x = (-e * offsets[i].y - b) / (2 * d);
		coefs[i] = { a, b, c, d, e, f };

	}
	return {offsets, coefs};
}


}
