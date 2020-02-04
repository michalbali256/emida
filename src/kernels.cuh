#pragma once

#include <cstdint>

#include "common.hpp"

namespace emida
{

template<typename T, typename RES>
void run_cross_corr(const T* pic_a, const T* pic_b, RES* res, int cols, int rows);

template<typename T>
void run_hanning(T* pic, const T* hanning_x, const T* hanning_y, int cols, int rows);

}