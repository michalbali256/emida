
#include "gpu_offset.hpp"

namespace emida {

template<typename T>
inline std::vector<vec2<T>> get_offset(T* pic, T* temp, vec2<size_t> size, vec2<size_t> cross_size, size_t b_size)
{
	gpu_offset<T> offs(size, cross_size, b_size);
	offs.allocate_memory();
	return offs.get_offset(pic, temp);
}

template<typename T>
inline std::vector<vec2<T>> get_offset(T* pic, T* temp, vec2<size_t> size, size_t b_size)
{
	return get_offset(pic, temp, size, 2 * size - 1, b_size);
}

template<typename T>
inline std::vector<vec2<size_t>> get_maxarg(const T* cu_data, size2_t size, size_t batch_size)
{
	gpu_offset<T> offs(size + 1 / 2, size, batch_size);
	offs.allocate_memory();
	return offs.get_maxarg(cu_data);
}

}