#pragma once

#include "cuda.h"

namespace emida
{

template <typename T>
__device__ T* shared_memory_proxy()
{
	extern __shared__ unsigned char memory[];
	return reinterpret_cast<T*>(memory);
}


}