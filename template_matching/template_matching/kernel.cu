
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <bitmap_image.hpp>
#include <iostream>

using namespace std;

int main()
{
	bitmap_image mainImageBmp("collection_coin.bmp");
	bitmap_image templateImageBmp("template.bmp");

	if (!(mainImageBmp) || !(templateImageBmp)) {
		printf("Error - Failed to open: input.bmp\n");
		return 1;
	}

	const unsigned int mainHeight = mainImageBmp.height();
	const unsigned int mainWidth = mainImageBmp.width();
	const unsigned int templateHeight = templateImageBmp.height();
	const unsigned int templateWidth = templateImageBmp.width();

	const unsigned int total_num_of_pixels_main = 3 * mainHeight * mainWidth;
	const unsigned int total_num_of_pixels_template = 3 * templateHeight * templateWidth;


	char *mainMatrix = new char[total_num_of_pixels_main];
	char *templateMatrix = new char[total_num_of_pixels_template];


	system("pause");
	return 0;



}

void extract_image_matrix(bitmap_image imageBmp, char *imageMatrix, unsigned int width, unsigned int height)
{
	for (int row = 0; row < height; row++) {
		for (int col = 0; col < width; col++) {
			char proper_index_r = row * width + col;
			char proper_index_g = proper_index_r + sizeof(char);
			char proper_index_b = proper_index_r + 2 * sizeof(char);

			imageMatrix[proper_index_r] = char(imageBmp.get_pixel(row, col).red);
			imageMatrix[proper_index_g] = char(imageBmp.get_pixel(row, col).green);
			imageMatrix[proper_index_b] = char(imageBmp.get_pixel(row, col).blue);
		}
	}
}