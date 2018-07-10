#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>
#include <chrono>
#include <math_functions.h>
#include <bitmap_image.hpp>
#include <cufft.h>

#define errorHandler(stmt)																					\
	do {																									\
		cudaError_t err = stmt;																				\
		if (err != cudaSuccess) {																			\
			printf("[ERROR] Failed to run stmt %d, error body: %s\n", __LINE__, cudaGetErrorString(err));	\
			return -1; }																 					\
	} while (0)																								\

#define M_PI 3.14159265
#define BLOCK_SIZE_X 32
#define BLOCK_SIZE_Y 32
#define BLOCK_SIZE 1024

using namespace std;

int initiate_parallel_template_matching(bitmap_image, bitmap_image);
void initiate_serial_template_matching(bitmap_image, bitmap_image);
void device_query();
void extract_array(unsigned char*, unsigned int, bitmap_image);
int get_number_of_occurances(int * arr, unsigned int size);


int main()
{
	bitmap_image main_image("Input Files/collection.bmp");
	bitmap_image template_image("Input Files/collection_coin.bmp");

	initiate_parallel_template_matching(main_image, template_image);
	wcout << "\n ------- ******************* ------- \n";
	// initiate_serial_template_matching(main_image, template_image);
	// device_query();
	system("pause");
	return 0;
}

int	initiate_parallel_template_matching(bitmap_image main_image, bitmap_image template_image)
{
	// Get sizes
	int main_width = main_image.width();
	int main_height = main_image.height();
	int main_size = main_width * main_height;
	int template_width = template_image.width();
	int template_height = template_image.height();
	int template_size = template_width * template_height;
	int height_difference = main_height - template_height;
	int width_difference = main_width - template_width;
	int mse_array_size = (height_difference + 1) * (width_difference + 1);

	unsigned char* h_main_image = new unsigned char[3 * main_size];

	for (int col = 0; col < main_width; col++) {
		for (int row = 0; row < main_height; row++) {
			rgb_t colors;

			main_image.get_pixel(col, row, colors);
			h_main_image[(row * main_width + col) * 3 + 0] = colors.red;
			h_main_image[(row * main_width + col) * 3 + 1] = colors.green;
			h_main_image[(row * main_width + col) * 3 + 2] = colors.blue;
		}
	}

	unsigned char* h_template_image = new unsigned char[3 * template_size];

	for (int col = 0; col < template_width; col++) {
		for (int row = 0; row < template_height; row++) {
			rgb_t colors;

			template_image.get_pixel(col, row, colors);
			h_template_image[(row * template_width + col) * 3 + 0] = colors.red;
			h_template_image[(row * template_width + col) * 3 + 1] = colors.green;
			h_template_image[(row * template_width + col) * 3 + 2] = colors.blue;
		}
	}

	cufftComplex* h_main_signal = (cufftComplex *)malloc(sizeof(cufftComplex) * main_width * main_height * 3);
	cufftComplex* h_template_signal = (cufftComplex *)malloc(sizeof(cufftComplex) * template_width * template_height * 3);
	long unsigned int main_signal_size = main_width * main_height * 3;
	long unsigned int template_signal_size = template_width * template_height * 3;

	for (int y = 0; y < main_height; y++) {
		for (int x = 0; x < main_width; x++) {
			h_main_signal[(y * main_width + x) * 3 + 0].x = (double) h_main_image[(y * main_width + x) * 3 + 0];
			h_main_signal[(y * main_width + x) * 3 + 1].x = (double)h_main_image[(y * main_width + x) * 3 + 1];
			h_main_signal[(y * main_width + x) * 3 + 2].x = (double)h_main_image[(y * main_width + x) * 3 + 2];
			h_main_signal[(y * main_width + x) * 3 + 0].y = 0;
			h_main_signal[(y * main_width + x) * 3 + 1].y = 0;
			h_main_signal[(y * main_width + x) * 3 + 2].y = 0;
		}
	}

	for (int y = 0; y < template_height; y++) {
		for (int x = 0; x < template_width; x++) {
			h_template_signal[(y * template_width + x) * 3 + 0].x = (double)h_template_image[(y * template_width + x) * 3 + 0];
			h_template_signal[(y * template_width + x) * 3 + 1].x = (double)h_template_image[(y * template_width + x) * 3 + 1];
			h_template_signal[(y * template_width + x) * 3 + 2].x = (double)h_template_image[(y * template_width + x) * 3 + 2];
			h_template_signal[(y * template_width + x) * 3 + 0].y = 0;
			h_template_signal[(y * template_width + x) * 3 + 1].y = 0;
			h_template_signal[(y * template_width + x) * 3 + 2].y = 0;
		}
	}

	cufftComplex* d_main_signal;
	cufftComplex* d_template_signal;
	int main_memsize = sizeof(cufftComplex) * main_signal_size;
	int template_memsize = sizeof(cufftComplex) * template_signal_size;

	errorHandler(cudaMalloc((void**)&d_main_signal, main_memsize));
	errorHandler(cudaMalloc((void**)&d_template_signal, template_memsize));

	errorHandler(cudaMemcpy(d_main_signal, h_main_signal, main_memsize, cudaMemcpyHostToDevice));
	errorHandler(cudaMemcpy(d_template_signal, h_template_signal, template_memsize, cudaMemcpyHostToDevice));

	cufftHandle plan_main;
	cufftHandle plan_template;
	cufftPlan1d(&plan_main, main_signal_size, CUFFT_C2C, 1);
	cufftPlan1d(&plan_template, template_signal_size, CUFFT_C2C, 1);

	cufftExecC2C(plan_main, (cufftComplex *)d_main_signal, (cufftComplex *)d_main_signal, CUFFT_FORWARD);
	cufftExecC2C(plan_template, (cufftComplex *)d_template_signal, (cufftComplex *)d_template_signal, CUFFT_FORWARD);


	cuComplex* h_fft_main_signal = (cufftComplex*)malloc(sizeof(cufftComplex)* main_width * main_height * 3);
	cuComplex* h_fft_template_signal = (cufftComplex*)malloc(sizeof(cufftComplex)* template_width * template_height * 3);

	cudaMemcpy(h_fft_main_signal, d_main_signal, main_memsize, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_fft_template_signal, d_template_signal, template_memsize, cudaMemcpyDeviceToHost);

	// Complex conjugate
	for (int y = 0; y < main_height; y++) {
		for (int x = 0; x < main_width; x++) {
			h_fft_main_signal[(y * main_width + x) * 3 + 0].y *= -1;
			h_fft_main_signal[(y * main_width + x) * 3 + 1].y *= -1;
			h_fft_main_signal[(y * main_width + x) * 3 + 2].y *= -1;
		}
	}

	// Complex conjugate
	for (int y = 0; y < template_height; y++) {
		for (int x = 0; x < template_width; x++) {
			h_template_signal[(y * template_width + x) * 3 + 0].y *= -1;
			h_template_signal[(y * template_width + x) * 3 + 1].y *= -1;
			h_template_signal[(y * template_width + x) * 3 + 2].y *= -1;
		}
	}


	// cuComplex* h_correlation_signal = (cufftComplex*)malloc(sizeof())

	return EXIT_SUCCESS;
}

