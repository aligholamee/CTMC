#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>
#include <chrono>
#include <math_functions.h>
#include <bitmap_image.hpp>
#include <cufft.h>
#include <assert.h>

using namespace std;

#define errorHandler(stmt)																					\
	do {																									\
		cudaError_t err = stmt;																				\
		if (err != cudaSuccess) {																			\
			printf("[ERROR] Failed to run stmt %d, error body: %s\n", __LINE__, cudaGetErrorString(err));	\
			return -1; }																 					\
	} while (0)																								\

#define M_PI 3.14159265
 
#define BLOCK_SIZE 1024
typedef float2 Complex;

int initiate_parallel_template_matching(bitmap_image, bitmap_image);
static __device__ __host__ inline Complex ComplexAdd(Complex, Complex);
static __device__ __host__ inline Complex ComplexScale(Complex, float);
static __device__ __host__ inline Complex ComplexMul(Complex, Complex);
static __global__ void ComplexPointwiseMulAndScale(cufftComplex *, cufftComplex *, int, float);

inline bool
sdkCompareL2fe(const float* reference, const float* data,
	const unsigned int len, const float epsilon)
{
	assert(epsilon >= 0);

	float error = 0;
	float ref = 0;

	for (unsigned int i = 0; i < len; ++i) {

		float diff = reference[i] - data[i];
		error += diff * diff;
		ref += reference[i] * reference[i];
	}

	float normRef = sqrtf(ref);
	if (fabs(ref) < 1e-7) {
#ifdef _DEBUG
		std::cerr << "ERROR, reference l2-norm is 0\n";
#endif
		return false;
	}
	float normError = sqrtf(error);
	error = normError / normRef;
	bool result = error < epsilon;
#ifdef _DEBUG
	if (!result)
	{
		std::cerr << "ERROR, l2-norm error "
			<< error << " is greater than epsilon " << epsilon << "\n";
	}
#endif

	return result;
}

// Padding functions
int PadData(const cufftComplex *signal, cufftComplex **padded_signal, int signal_size,
	const cufftComplex *filter_kernel, cufftComplex **padded_filter_kernel, int filter_kernel_size);

// Filtering functions
void Convolve(const Complex *, int, const Complex *, int, Complex *);

int get_number_of_occurances(cufftComplex * arr, unsigned int size);

int main()
{
	bitmap_image main_image("Input Files/collection2.bmp");
	bitmap_image template_image("Input Files/collection_coin.bmp");

	initiate_parallel_template_matching(main_image, template_image);

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

	// Convert to grayscale
	main_image.convert_to_grayscale();
	template_image.convert_to_grayscale();

	unsigned char* h_main_image = new unsigned char[main_size];

	for (int col = 0; col < main_width; col++) {
		for (int row = 0; row < main_height; row++) {
			rgb_t color;

			main_image.get_pixel(col, row, color);
			h_main_image[row * main_width + col] = color.red;
		}
	}

	unsigned char* h_template_image = new unsigned char[template_size];

	for (int col = 0; col < template_width; col++) {
		for (int row = 0; row < template_height; row++) {
			rgb_t color;

			template_image.get_pixel(col, row, color);
			h_template_image[row * template_width + col] = color.red;
		}
	}

	cufftComplex* h_main_signal = (cufftComplex *)malloc(sizeof(cufftComplex) * main_width * main_height);
	cufftComplex* h_template_signal = (cufftComplex *)malloc(sizeof(cufftComplex) * template_width * template_height);
	int main_signal_size = main_width * main_height;
	int template_signal_size = template_width * template_height;

	for (int y = 0; y < main_height; y++) {
		for (int x = 0; x < main_width; x++) {
			h_main_signal[y * main_width + x].x = (float)h_main_image[y * main_width + x];
			h_main_signal[y * main_width + x].y = 0;
		}
	}


	for (int y = 0; y < template_height; y++) {
		for (int x = 0; x < template_width; x++) {
			h_template_signal[y * template_width + x].x = (float)h_template_image[y * template_width + x];
			h_template_signal[y * template_width + x].y = 0;
		}
	}

	cufftComplex* d_main_signal;
	cufftComplex* d_template_signal;
	cufftComplex* d_main_signal_out;
	cufftComplex* d_template_signal_out;
	cufftComplex* d_inversed;


	// Pad image signals
	cufftComplex *h_padded_main_signal;
	cufftComplex *h_padded_template_signal;


	int NEW_SIZE = PadData(h_main_signal, &h_padded_main_signal, main_signal_size, h_template_signal, &h_padded_template_signal, template_signal_size);

	errorHandler(cudaMalloc((void**)&d_main_signal, sizeof(cufftComplex) * NEW_SIZE));
	errorHandler(cudaMalloc((void**)&d_template_signal, sizeof(cufftComplex) * NEW_SIZE));
	errorHandler(cudaMalloc((void**)&d_main_signal_out, sizeof(cufftComplex) * NEW_SIZE));
	errorHandler(cudaMalloc((void**)&d_template_signal_out, sizeof(cufftComplex) * NEW_SIZE));
	errorHandler(cudaMalloc((void**)&d_inversed, sizeof(cufftComplex) * NEW_SIZE));
	errorHandler(cudaMemcpy(d_main_signal, h_padded_main_signal, sizeof(cufftComplex) * NEW_SIZE, cudaMemcpyHostToDevice));
	errorHandler(cudaMemcpy(d_template_signal, h_padded_template_signal, sizeof(cufftComplex) * NEW_SIZE, cudaMemcpyHostToDevice));

	// Plan for 2 CUFFT_FORWARDs
	cufftHandle plan_main;
	cufftHandle plan_template;
	cufftPlan2d(&plan_main, int(sqrt(NEW_SIZE)), int(sqrt(NEW_SIZE)), CUFFT_C2C);
	cufftPlan2d(&plan_template, int(sqrt(NEW_SIZE)), int(sqrt(NEW_SIZE)), CUFFT_C2C);

	// Perform forward FFT
	cufftExecC2C(plan_main, (cufftComplex *)d_main_signal, (cufftComplex *)d_main_signal_out, CUFFT_FORWARD);
	cufftExecC2C(plan_template, (cufftComplex *)d_template_signal, (cufftComplex *)d_template_signal_out, CUFFT_FORWARD);

	// Multiply the coefficients together and normalize the result
	printf("Launching ComplexPointwiseMulAndScale<<< >>>\n");
	dim3 gridDimensions((unsigned int)(ceil(NEW_SIZE / (float)BLOCK_SIZE)), 1, 1);
	dim3 blockDimensions(BLOCK_SIZE, 1, 1);

	ComplexPointwiseMulAndScale << <gridDimensions, blockDimensions >> >((cufftComplex *)d_main_signal_out, (cufftComplex *)d_template_signal_out, NEW_SIZE, 1.0f / NEW_SIZE);
	cout << "Successfully completed complex pointwise mul and scale" << endl;
	errorHandler(cudaGetLastError());

	// Perform the inverse fft on the main signal
	cufftExecC2C(plan_main, (cufftComplex *)d_main_signal_out, (cufftComplex *)d_inversed, CUFFT_INVERSE);

	// Copy data back to host
	cufftComplex * h_correlation_signal;
	h_correlation_signal = h_padded_main_signal;
	errorHandler(cudaMemcpy(h_correlation_signal, d_inversed, sizeof(cufftComplex) * NEW_SIZE, cudaMemcpyDeviceToHost));

	for (int i = 0; i < NEW_SIZE; i++) {
		h_correlation_signal[i].x = abs(h_correlation_signal[i].x);
		h_correlation_signal[i].y = abs(h_correlation_signal[i].y);
	}
	
	ofstream convolveResult;
	convolveResult.open("convRes.txt");

	for (int i = 0; i < NEW_SIZE; i++) {
		convolveResult << "(" << h_correlation_signal[i].x << ", " << h_correlation_signal[i].y << ")" << endl;
	}

	convolveResult.close();

	/*cufftComplex* h_convolved_on_cpu;
	h_convolved_on_cpu = (Complex*)malloc(sizeof(Complex) * main_signal_size);

	/*
	// Convolve on the host
	Convolve(h_main_signal, main_signal_size, h_template_signal,
		template_signal_size, h_convolved_on_cpu);

	// Compare CPU and GPU result
	bool bTestResult = sdkCompareL2fe((float *)h_convolved_on_cpu,
		(float *)h_correlation_signal, 2 * main_signal_size,
		1e-5f);

	printf("\nvalue of TestResult %d\n", bTestResult);

	*/
	get_number_of_occurances(h_correlation_signal, NEW_SIZE);


	// Cancel plans :))))
	cufftDestroy(plan_main);
	cufftDestroy(plan_template);

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
static __global__ void ComplexPointwiseMulAndScale(cufftComplex *a, cufftComplex *b, int size, float scale)
{
	const int numThreads = blockDim.x * gridDim.x;
	const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
	for (int i = threadID; i < size; i += numThreads)
	{
		a[i] = ComplexScale(ComplexMul(a[i], b[i]), scale);
	}
}

int get_number_of_occurances(cufftComplex * arr, unsigned int size)
{
	cufftComplex max = arr[0];
	int num_of_occurs = 0;

	for (unsigned int i = 1; i < size; i++) {
		if (arr[i].x > max.x) {
			num_of_occurs = 1;
			max = arr[i];
		}

		if (arr[i].x == max.x)
			num_of_occurs++;
	}

	wcout << "[Found Maximum]: " << max.x << endl;
	wcout << "[Number of Occurances]: " << num_of_occurs << endl;

	return num_of_occurs;
}

////////////////////////////////////////////////////////////////////////////////
// Filtering operations - Computing Convolution on the host
////////////////////////////////////////////////////////////////////////////////
void Convolve(const Complex *signal, int signal_size,
	const Complex *filter_kernel, int filter_kernel_size,
	Complex *filtered_signal)
{
	int minRadius = filter_kernel_size / 2;
	int maxRadius = filter_kernel_size - minRadius;

	// Loop over output element indices
	for (int i = 0; i < signal_size; ++i)
	{
		if (i % 1000 == 0)
			cout << ".";

		filtered_signal[i].x = filtered_signal[i].y = 0;

		// Loop over convolution indices
		for (int j = -maxRadius + 1; j <= minRadius; ++j)
		{
			int k = i + j;

			if (k >= 0 && k < signal_size)
			{
				filtered_signal[i] = ComplexAdd(filtered_signal[i], ComplexMul(signal[k], filter_kernel[minRadius - j]));
			}
		}
	}
}