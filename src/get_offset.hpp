
#include "gpu_offset.hpp"

namespace emida {


template<typename T>
inline std::vector<vec2<T>> get_offset(T* pic, T* temp, vec2<size_t> size, vec2<size_t> cross_size, cross_policy c_policy = CROSS_POLICY_BRUTE)
{
	std::vector<size2_t> begins = { {0,0} };
	gpu_offset<T, T> offs(size, &begins, size, cross_size, c_policy);
	offs.allocate_memory(temp);
	return offs.get_offset(pic).offsets;
}

template<typename T>
inline std::vector<vec2<T>> get_offset(T* pic, T* temp, vec2<size_t> size, size_t b_size)
{
	return get_offset(pic, temp, size, 2 * size - 1);
}

template<typename T>
inline std::vector<vec2<size_t>> get_maxarg(const T* cu_data, size2_t size, size_t batch_size)
{
	std::vector<size2_t> begins(batch_size);
	gpu_offset<T, T> offs({}, &begins, size + 1 / 2, size);
	offs.allocate_memory(nullptr);
	return offs.get_maxarg(cu_data);
}

}