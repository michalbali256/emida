#include <gtest/gtest.h>

#include "gpu_offset.hpp"
#include "host_helpers.hpp"

using namespace emida;

TEST(fft, test)
{
	std::vector<double> in
	{
		1, 2, 3, 1, 0, 0, 0, 0, 0, 0,
		4, 5, 6, 1, 0, 0, 0, 0, 0, 0,
		7, 8, 9, 1, 0, 0, 0, 0, 0, 0,
		7, 8, 9, 1, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0
	};

	std::vector<double> in_temp
	{
		6, 3, 2, 1, 0, 0, 0, 0, 0, 0,
		12, 23, 3, 1, 0, 0, 0, 0, 0, 0,
		1, 8, 9, 1, 0, 0, 0, 0, 0, 0,
		7, 8, 9, 1, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0
	};

	size2_t slice_size{ 4,4 };
	std::vector<size2_t> begins{ {0,0} };
	gpu_offset<double, uint16_t> offs({ 0,0 }, &begins, slice_size, slice_size * 2 - 1, CROSS_POLICY_FFT);
	offs.allocate_memory();
	copy_to_device(in, offs.get_cu_pic());
	copy_to_device(in_temp, offs.get_cu_temp());

	offs.cross_corr_fft();
}
