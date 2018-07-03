
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>

using namespace std;

struct BITMAP {
	int width; 
	int height;
	unsigned char header[54];
	unsigned char *pixels;
	int size;
};

int main()
{
	
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



}