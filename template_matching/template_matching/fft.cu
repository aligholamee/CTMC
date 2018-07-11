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
typedef float2 Complex;

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
			h_main_signal[(y * main_width + x) * 3 + 0].x = (double)h_main_image[(y * main_width + x) * 3 + 0];
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
	cufftComplex* d_main_signal_out;
	cufftComplex* d_template_signal_out;

	int main_memsize = sizeof(cufftComplex) * main_signal_size;
	int template_memsize = sizeof(cufftComplex) * template_signal_size;

	// Pad image signals
	cufftComplex *h_padded_main_signal;
	cufftComplex *h_padded_template_signal;

	int NEW_SIZE = PadData(h_main_signal, &h_padded_main_signal, main_signal_size, h_template_signal, &h_padded_template_signal, template_signal_size);



	errorHandler(cudaMalloc((void**)&d_main_signal, NEW_SIZE));
	errorHandler(cudaMalloc((void**)&d_template_signal, NEW_SIZE));
	errorHandler(cudaMalloc((void**)&d_main_signal_out, NEW_SIZE));
	errorHandler(cudaMalloc((void**)&d_template_signal_out, NEW_SIZE));
	errorHandler(cudaMemcpy(d_main_signal, h_padded_main_signal, NEW_SIZE, cudaMemcpyHostToDevice));
	errorHandler(cudaMemcpy(d_template_signal, h_padded_template_signal, NEW_SIZE, cudaMemcpyHostToDevice));

	// Plan for 2 CUFFT_FORWARDs :)))
	cufftHandle plan_main;
	cufftHandle plan_template;
	cufftPlan1d(&plan_main, NEW_SIZE, CUFFT_C2C, 1);
	cufftPlan1d(&plan_template, NEW_SIZE, CUFFT_C2C, 1);

	// Perform forward FFT
	cufftExecC2C(plan_main, (cufftComplex *)d_main_signal, (cufftComplex *)d_main_signal, CUFFT_FORWARD);
	cufftExecC2C(plan_template, (cufftComplex *)d_template_signal, (cufftComplex *)d_template_signal, CUFFT_FORWARD);

	// Copy fft results to another location on device
	errorHandler(cudaMemcpy(d_main_signal_out, d_main_signal, NEW_SIZE, cudaMemcpyDeviceToDevice));
	errorHandler(cudaMemcpy(d_template_signal_out, d_template_signal, NEW_SIZE, cudaMemcpyDeviceToDevice));

	//Multiply the coefficients together and normalize the result
	printf("Launching ComplexPointwiseMulAndScale<<< >>>\n");
	ComplexPointwiseMulAndScale((cufftComplex *)d_main_signal_out, (cufftComplex *)d_template_signal_out, NEW_SIZE, 1.0f / NEW_SIZE);

	// Perform the inverse fft on the main signal
	cufftExecC2C(plan_main, (cufftComplex *)d_main_signal, (cufftComplex *)d_main_signal, CUFFT_INVERSE);

	// Copy data back to host
	cufftComplex * h_correlation_signal;
	h_correlation_signal = h_padded_main_signal;
	errorHandler(cudaMemcpy(h_correlation_signal, d_main_signal, NEW_SIZE, cudaMemcpyDeviceToHost));

	// Free allocated memory
	errorHandler(cudaFree(d_main_signal));
	errorHandler(cudaFree(d_template_signal));
	errorHandler(cudaFree(d_main_signal_out));
	errorHandler(cudaFree(d_template_signal_out));
	free(h_main_image);
	free(h_template_image);
	free(h_main_signal);
	free(h_template_signal);
	free(h_padded_main_signal);
	free(h_padded_template_signal);
	free(h_correlation_signal);
	return EXIT_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////////
// Function for padding original data
//////////////////////////////////////////////////////////////////////////////////
int PadData(const cufftComplex *signal, cufftComplex **padded_signal, int signal_size,
	const cufftComplex *filter_kernel, cufftComplex **padded_filter_kernel, int filter_kernel_size)
{
	int minRadius = filter_kernel_size / 2;
	int maxRadius = filter_kernel_size - minRadius;
	int new_size = signal_size + maxRadius;

	// Pad signal
	cufftComplex *new_data = (cufftComplex *)malloc(sizeof(cufftComplex) * new_size);
	memcpy(new_data + 0, signal, signal_size * sizeof(cufftComplex));
	memset(new_data + signal_size, 0, (new_size - signal_size) * sizeof(cufftComplex));
	*padded_signal = new_data;

	// Pad filter
	new_data = (cufftComplex *)malloc(sizeof(cufftComplex) * new_size);
	memcpy(new_data + 0, filter_kernel + minRadius, maxRadius * sizeof(cufftComplex));
	memset(new_data + maxRadius, 0, (new_size - filter_kernel_size) * sizeof(cufftComplex));
	memcpy(new_data + new_size - minRadius, filter_kernel, minRadius * sizeof(cufftComplex));
	*padded_filter_kernel = new_data;

	return new_size;
}


////////////////////////////////////////////////////////////////////////////////
// Complex operations
////////////////////////////////////////////////////////////////////////////////

// Complex addition
static __device__ __host__ inline Complex ComplexAdd(Complex a, Complex b)
{
	Complex c;
	c.x = a.x + b.x;
	c.y = a.y + b.y;
	return c;
}

// Complex scale
static __device__ __host__ inline Complex ComplexScale(Complex a, float s)
{
	Complex c;
	c.x = s * a.x;
	c.y = s * a.y;
	return c;
}

// Complex multiplication
static __device__ __host__ inline Complex ComplexMul(Complex a, Complex b)
{
	Complex c;
	c.x = a.x * b.x - a.y * b.y;
	c.y = a.x * b.y + a.y * b.x;
	return c;
}
// Complex pointwise multiplication
static __global__ void ComplexPointwiseMulAndScale(cufftComplex *a, cufftComplex *b, int size, int scale)
{
	const int numThreads = blockDim.x * gridDim.x;
	const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
	for (int i = threadID; i < size; i += numThreads)
	{
		a[i] = ComplexScale(ComplexMul(a[i], b[i]), scale);
	}
}

