
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>
#include <iostream>

#define M_PI 3.14159265
using namespace std;

struct BITMAP {
	int width; 
	int height;
	unsigned char header[54];
	unsigned char *pixels;
	int size;
};

BITMAP read_bitmap_image(string file_name);
BITMAP rotate_bitmap_image(BITMAP image, double degree);
void save_bitmap_image(string file_name, BITMAP image);

int main()
{
	BITMAP mainImage = read_bitmap_image("collection_coin.bmp");
	mainImage = rotate_bitmap_image(mainImage, 270);
	save_bitmap_image("rotatedBitMap.bmp", mainImage);

	system("pause");
	return 0;
}

BITMAP read_bitmap_image(string file_name)
{
	BITMAP image;
	int i;
	string file_path = "Input Files/" + file_name;
	FILE *f = fopen(file_path.c_str(), "rb");
	fread(image.header, sizeof(unsigned char), 54, f);

	image.width = *(int *)&image.header[18];
	image.height = *(int *)&image.header[22];

	// 3 Bytes per pixel
	image.size = 3 * image.width * image.height;

	image.pixels = new unsigned char[image.size];
	fread(image.pixels, sizeof(unsigned char), image.size, f);
	fclose(f);

	for (i = 0; i < image.size; i += 3) {
		unsigned char tmp = image.pixels[i];
		image.pixels[i] = image.pixels[i + 2];
		image.pixels[i + 2] = tmp;
	}

	return image;
}

BITMAP rotate_bitmap_image(BITMAP image, double degree)
{
	BITMAP rotated = image;
	unsigned char *pixels = new unsigned char[image.size];
	double radians = (degree * M_PI) / 180;
	int sinf = (int)sin(radians);
	int cosf = (int)cos(radians);

	double x0 = 0.5 * (image.width - 1); 
	double y0 = 0.5 * (image.height - 1);

	for (int x = 0; x < image.width; x++) {
		for (int y = 0; y < image.height; y++) {
			long double a = x - x0;
			long double b = y - y0;
			int xx = (int)(+a * cosf - b * sinf + x0);
			int yy = (int)(+a * sinf + b * cosf + y0);

			if (xx >= 0 && xx < image.width && yy >= 0 && yy < image.height) {
				pixels[(y * image.height + x) * 3 + 0] = image.pixels[(yy * image.height + xx) * 3 + 0];
				pixels[(y * image.height + x) * 3 + 1] = image.pixels[(yy * image.height + xx) * 3 + 1];
				pixels[(y * image.height + x) * 3 + 2] = image.pixels[(yy * image.height + xx) * 3 + 2];
			}
		}
	}

	rotated.pixels = pixels;
	return rotated;
}

void save_bitmap_image(string file_name, BITMAP image)
{
	string file_path = "Output Files/" + file_name;
	FILE *out = fopen(file_path.c_str(), "wb");
	fwrite(image.header, sizeof(unsigned char), 54, out);

	int i;
	unsigned char tmp;
	for (i = 0; i < image.size; i += 3) {
		tmp = image.pixels[i];
		image.pixels[i] = image.pixels[i + 2];
		image.pixels[i + 2] = tmp;
	}

	fwrite(image.pixels, sizeof(unsigned char), image.size, out);
	fclose(out);
}

