
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "kernels.cuh"
namespace emida {

template<typename T>
__global__ void extract_neighbors(const T* data, const vec2<size_t>* max_i, T* neighbors, int s, size2_t src_size, size_t batch_size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= batch_size)
		return;

	int r = (s - 1) / 2;

	size2_t max_pos = max_i[idx];

	for (int i = 0; i < s; ++i)
	{
		for (int j = 0; j < s; ++j)
		{
			int from_x = (int)max_pos.x - r + i;
			int from_y = (int)max_pos.y - r + j;

			from_x = max(from_x, 0);
			from_x = min(from_x, (int)src_size.x);

			from_y = max(from_y, 0);
			from_y = min(from_y, (int)src_size.y);

			neighbors[idx * s * s + j * s + i] = data[idx * src_size.area() + from_y * src_size.x + from_x];
		}
	}
}

template<typename T>
void run_extract_neighbors(const T* data, const vec2<size_t>* max_i, T* neighbors, int s, size2_t src_size, size_t batch_size)
{
	size_t block_size = 128;
	size_t grid_size = div_up(batch_size, block_size);
	extract_neighbors<T> << <grid_size, block_size >> > (data, max_i, neighbors, s, src_size, batch_size);
}

template void run_extract_neighbors<double>(const double* data, const vec2<size_t>* max_i, double* neighbors, int s, size2_t src_size, size_t batch_size);
template void run_extract_neighbors<float>(const float* data, const vec2<size_t>* max_i, float* neighbors, int s, size2_t src_size, size_t batch_size);

}