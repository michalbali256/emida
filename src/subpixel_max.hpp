
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
template <typename T>
void multiply(const T* a, const T* b, T* c, size_t m, size_t n, size_t k)
{
	for (size_t x = 0; x < n; ++x)
	{
		for (size_t y = 0; y < m; ++y)
		{
			T sum = 0;
			for (size_t i = 0; i < k; ++i)
			{
				sum += a[y * k + i] * b[i * n + x];
			}
			c[y * n + x] = sum;
		}
	}
}

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
	std::array<T, 6> coef;
	multiply<T>(lstsq_matrix<T, s>::mat.data(), pic, coef.data(), 6, 1, s * s);
	return coef;
}

template<typename T, int s>
vec2<T> subpixel_max_serial(const T* pic)
{
	//coefficients of quadratic function of neighborhood of maximum
	//f(x,y) = a + bx + cy + dx^2 + exy + fy^2
	auto [a, b, c, d, e, f] = subpixel_max_coefs<T, s>(pic);

	//now get the maximum of that function. Partial derivations:
	//from partial derivation by x : 2dx + ey + b = 0
	//from partial derivation by y : ex + 2fy + c = 0
	//solve the 2 equations:

	vec2<T> ret;
	ret.y = (b * e - 2 * c * d) / (4 * f * d - e * e);
	ret.x = (-e * ret.y - b) / (2 * d);

	return ret;
}


}
