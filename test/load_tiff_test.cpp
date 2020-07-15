#include <gtest/gtest.h>

#include "load_tiff.hpp"

using namespace emida;

TEST(load_tiff, load)
{
	//TIFF* tif = TIFFOpen("test/res/text_5x5.tif", "r");
	size2_t size { 873, 873 };
	std::vector<uint16_t> raster(size.area());
	std::string file_name = "test/res/INITIAL_x0y0.tif";
	load_tiff(file_name, raster.data(), size);

	TIFF* tif = TIFFOpen(file_name.c_str(), "r");
	ASSERT_NE(tif, nullptr);

	uint32 w, h;
	size_t npixels;
	std::vector<uint32> rasterRGB(size.area());

	TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &w);
	TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &h);
	npixels = w * h;
	
	int read_err = TIFFReadRGBAImage(tif, w, h, rasterRGB.data());
	ASSERT_EQ(read_err, 1);
	
	std::vector<uint16_t> greyscale(rasterRGB.size());

	for (size_t i = 0; i < rasterRGB.size(); ++i)
	{
		size_t x = i % w;
		size_t y = i / w;
		uint32_t pixel = rasterRGB[(h-y-1)*w + x];
		uint32_t r = TIFFGetR(pixel);
		uint32_t g = TIFFGetG(pixel);
		uint32_t b = TIFFGetB(pixel);
		ASSERT_EQ(r, g);
		ASSERT_EQ(g, b);
		uint16_t grey = raster[i] >> 8;
		ASSERT_EQ(r, grey) << " at " << i << " x = " << i % w << " y = " << i / w << "\n";

	}

	TIFFClose(tif);
	
	int i = 0;
}