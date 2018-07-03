
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <bitmap_image.hpp>
#include <iostream>

using namespace std;

void extract_image_matrix(bitmap_image imageBmp, size_t *imageMatrix, size_t width, size_t height);
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

	for (size_t row = 0; row < mainHeight; row++) {
		for (size_t col = 0; col < mainWidth; col++) {
			cout << "red: " << int(mainImageBmp.get_pixel(row, col).red);
			cout << "green: " << int(mainImageBmp.get_pixel(row, col).green);
			cout << "blue: " << int(mainImageBmp.get_pixel(row, col).blue);
		}
		cout << endl;
	}
	//extract_image_matrix(mainImageBmp, mainMatrix, mainWidth, mainHeight);
	//print_image_pixels(mainMatrix, mainWidth, mainHeight);

	system("pause");
	return 0;
}

void extract_image_matrix(bitmap_image imageBmp, size_t *imageMatrix, size_t width, size_t height)
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

void print_image_pixels(size_t* imageMatrix, size_t width, size_t height)
{
	for (size_t row = 0; row < height; row++) {
		for (size_t col = 0; col < width; col++)
			cout << imageMatrix[row * (width * 3) + col*3] << " ";
		cout << endl;
	}
}
