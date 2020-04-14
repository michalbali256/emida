#pragma once
#include <cassert>
#include <cmath>

#include "tiffio.h"

#include "common.hpp"

namespace emida
{

template <typename T>
void draw_vector(T* pic, size2_t size, vec2<double> vector, size2_t vector_pos, T color)
{
	for (double d = 0; d < 1; d += 0.05)
	{
		size2_t point = { vector_pos.x + (int)round(d * vector.x), vector_pos.y + (int)round(d * vector.y) };
		if(point.x >= 0 && point.x < size.x && point.y >= 0 && point.y < size.y)
			pic[point.pos(size.x)] = color;
	}
}


template <typename T>
void draw_tiff(T * pic, size2_t size, std::string file_name, std::vector<vec2<double>> vectors, std::vector<size2_t> vectors_pos)
{
	assert(vectors.size() == vectors_pos.size());

	std::vector<uint32_t> rgb_picture(size.area());

	for (size_t i = 0; i < size.area(); i++)
	{
		uint32_t val = pic[i] >> 8;
		rgb_picture[i] = 0xFF000000 | val | (val << 8) | (val << 16);
	}

	for (size_t i = 0; i < vectors.size(); ++i)
	{
		draw_vector(rgb_picture.data(), size, vectors[i], vectors_pos[i], 0xFF0000FF);
	}

	TIFF * tif = TIFFOpen(file_name.c_str(), "w");
	TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, (uint32_t) size.x);
	TIFFSetField(tif, TIFFTAG_IMAGELENGTH, (uint32_t)size.y);
	TIFFSetField(tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
	TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, 4);
	TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, 8);
	TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_RGB);
	
	

	
	for (uint32_t y = 0; y < size.y; ++y)
	{
		TIFFWriteScanline(tif, rgb_picture.data() + y * size.x, y);
	}
	TIFFClose(tif);
}

}