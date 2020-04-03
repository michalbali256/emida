#include <string>

#include "tiffio.h"

#include "common.hpp"

namespace emida
{

inline void load_tiff(const std::string & file_name, uint16_t * raster, vec2<size_t> size)
{
	//error checking?
	TIFF* tif = TIFFOpen(file_name.c_str(), "r");
	uint32 w, h;
	TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &w);
	TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &h);
	if (w != size.x || h != size.y)
		throw std::runtime_error("The size of " + file_name + " is different than specified size.");
	uint16 orient;
	TIFFGetField(tif, TIFFTAG_ORIENTATION, &orient);
	if (orient != ORIENTATION_TOPLEFT)
	{
		throw std::runtime_error("The orientation of '" + file_name +
			"' is wrong, only able to process row 0 TOP, col 0 LEFT ortientation.");
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
}


}