
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "kernels.cuh"
namespace emida {

template<typename T, typename pos_policy>
__global__ void extract_neighbors(const T* data, const size2_t* max_i, T* neighbors, int s, size2_t src_size, size_t batch_size)
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
			int from_x_i = (int)max_pos.x - r + i;
			int from_y_i = (int)max_pos.y - r + j;

			esize_t from_x = max(from_x_i, 0);
			from_x = min(from_x, src_size.x);

			esize_t from_y = max(from_y_i, 0);
			from_y = min(from_y, src_size.y);
			

			neighbors[idx * s * s + j * s + i] = data[pos_policy::pos(idx * src_size.area() + from_y * src_size.x + from_x, idx, {from_x, from_y}, src_size)];
		}
	}
}

template<typename T, typename pos_policy>
void run_extract_neighbors(const T* data, const size2_t* max_i, T* neighbors, int s, size2_t src_size, esize_t batch_size)
{
	esize_t block_size = 128;
	esize_t grid_size = div_up(batch_size, block_size);
	extract_neighbors<T, pos_policy> <<<grid_size, block_size >>> (data, max_i, neighbors, s, src_size, batch_size);
}

template void run_extract_neighbors<double, cross_res_pos_policy_id>(const double* data, const size2_t* max_i, double* neighbors, int s, size2_t src_size, esize_t batch_size);
template void run_extract_neighbors<double, cross_res_pos_policy_fft>(const double* data, const size2_t* max_i, double* neighbors, int s, size2_t src_size, esize_t batch_size);
template void run_extract_neighbors<float, cross_res_pos_policy_id>(const float* data, const size2_t* max_i, float* neighbors, int s, size2_t src_size, esize_t batch_size);
template void run_extract_neighbors<float, cross_res_pos_policy_fft>(const float* data, const size2_t* max_i, float* neighbors, int s, size2_t src_size, esize_t batch_size);

}