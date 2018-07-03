
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <bitmap_image.hpp>
#include <iostream>

using namespace std;

void extract_rgb_image_matrix(size_t *imageMatrix, bitmap_image imageBmp, size_t width, size_t height);
void rgb_to_grayscale(size_t* grayScaled, size_t *imageMatrix, size_t width, size_t height);
void print_image_pixels(size_t* imageMatrix, size_t width, size_t height);

int main()
{
	bitmap_image mainImageBmp("collection_coin.bmp");

	if (!(mainImageBmp)) {
		printf("Error - Failed to open: input.bmp\n");
		return 1;
	}

	size_t mainHeight = mainImageBmp.height();
	size_t mainWidth = mainImageBmp.width();
	 
	size_t total_num_of_pixels_main = mainHeight * mainWidth;

	size_t *mainMatrix = new size_t[3 * total_num_of_pixels_main];

	extract_rgb_image_matrix(mainMatrix, mainImageBmp, mainWidth, mainHeight);

	size_t *grayScaled = new size_t[total_num_of_pixels_main];
	rgb_to_grayscale(grayScaled, mainMatrix, mainWidth, mainHeight);

	print_image_pixels(grayScaled, mainWidth, mainHeight);

	system("pause");
	return 0;
}

void extract_rgb_image_matrix(size_t *imageMatrix, bitmap_image imageBmp, size_t width, size_t height)
{
	for (size_t row = 0; row < height; row++) {
		for (size_t col = 0; col < width; col++) {
			size_t proper_index_r = row * (width * 3) + col * 3;
			size_t proper_index_g = proper_index_r + 1;
			size_t proper_index_b = proper_index_r + 2;

			imageMatrix[proper_index_r] = imageBmp.get_pixel(row, col).red;
			imageMatrix[proper_index_g] = imageBmp.get_pixel(row, col).green;
			imageMatrix[proper_index_b] = imageBmp.get_pixel(row, col).blue;
		}
	}
}

void rgb_to_grayscale(size_t* grayScaled, size_t *imageMatrix, size_t width, size_t height)
{
	for (size_t row = 0; row < height; row++) {
		for (size_t col = 0; col < width; col++) {
			int r_offset = row * width * 3 + col * 3;
			int g_offset = r_offset + 1;
			int b_offset = r_offset + 2;

			int grayScaledPixel = (int)(0.299 * (float)imageMatrix[r_offset] + 0.587 * (float)imageMatrix[g_offset] + 0.114 * (float)imageMatrix[b_offset]);
			grayScaled[row * width + col] = grayScaledPixel;
		}
	}
}

void print_image_pixels(size_t* imageMatrix, size_t width, size_t height)
{
	for (size_t row = 0; row < height; row++) {
		for (size_t col = 0; col < width; col++)
			cout << imageMatrix[row * width + col] << " ";
		cout << endl;
	}
}
