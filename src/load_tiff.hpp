#pragma once

#include <string>
#include <iostream>

#include "tiffio.h"

#include "common.hpp"

namespace emida
{

inline bool load_tiff(const std::string & file_name, uint16_t * raster, vec2<size_t> size)
{
	TIFF* tif = TIFFOpen(file_name.c_str(), "r");
	if (!tif)
		return false; //message is already written by libtiff
	
	uint32 w, h;
	TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &w);
	TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &h);
	if (w != size.x || h != size.y)
	{
		std::cerr << "The size of " + file_name + " is different than specified size " << size.x << "," << size.y << "\n";
		return false;
	}
	uint16 orient;
	TIFFGetField(tif, TIFFTAG_ORIENTATION, &orient);
	if (orient != ORIENTATION_TOPLEFT)
	{
		std::cerr << "The orientation of '" << file_name <<
			"' is wrong, only able to process row 0 TOP, col 0 LEFT ortientation.\n";
		return false;
	}

	size_t npixels = w * h;
	uint64* bc;
	
	TIFFGetField(tif, TIFFTAG_STRIPBYTECOUNTS, &bc);
	
	uint16_t* raster_end = raster + npixels;
	uint16_t* next_strip_dst = raster;

	for (uint32_t i = 0; i < TIFFNumberOfStrips(tif); ++i)
	{
		TIFFReadRawStrip(tif, i, next_strip_dst, bc[i]);
		next_strip_dst += bc[i] / sizeof(uint16_t);
	}
	TIFFClose(tif);

	return true;
}


}