#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>
#include <chrono>
#include <math_functions.h>
#include <bitmap_image.hpp>
#include <cufft.h>
#include <assert.h>
#include <cufftXt.h>

#define errorHandler(stmt)																					\
	do {																									\
		cudaError_t err = stmt;																				\
		if (err != cudaSuccess) {																			\
			printf("[ERROR] Failed to run stmt %d, error body: %s\n", __LINE__, cudaGetErrorString(err));	\
			return -1; }																 					\
	} while (0)																								\

inline bool
sdkCompareL2fe(const float *reference, const float *data,
	const unsigned int len, const float epsilon)
{
	assert(epsilon >= 0);

	float error = 0;
	float ref = 0;

	for (unsigned int i = 0; i < len; ++i)
	{

		float diff = reference[i] - data[i];
		error += diff * diff;
		ref += reference[i] * reference[i];
	}

	float normRef = sqrtf(ref);

	if (fabs(ref) < 1e-7)
	{
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

#define M_PI 3.14159265
#define BLOCK_SIZE_X 32
#define BLOCK_SIZE_Y 32
#define BLOCK_SIZE 1024
const int GPU_COUNT = 1;

using namespace std;

// Complex data type
typedef float2 Complex;

static __device__ __host__ inline Complex ComplexAdd(Complex, Complex);
static __device__ __host__ inline Complex ComplexScale(Complex, float);
static __device__ __host__ inline Complex ComplexMul(Complex, Complex);
static __global__ void ComplexPointwiseMulAndScale(cufftComplex *, cufftComplex *, int, float);

//Kernel for GPU
void multiplyCoefficient(cudaLibXtDesc *, cudaLibXtDesc *, int, float, int);

// Filtering functions
void Convolve(const Complex *, int, const Complex *, int, Complex *);

// Padding functions
int PadData(const Complex *, Complex **, int,
	const Complex *, Complex **, int);

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

	printf("\n[simpleCUFFT_MGPU] is starting...\n\n");

	int GPU_N;
	cudaGetDeviceCount(&GPU_N);

	if (GPU_N < GPU_COUNT)
	{
		printf("No. of GPU on node %d\n", GPU_N);
		printf("Two GPUs are required to run simpleCUFFT_MGPU sample code\n");
		return -1;
	}

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

	Complex* h_main_signal = (Complex *)malloc(sizeof(Complex) * main_width * main_height * 3);
	Complex* h_template_signal = (Complex *)malloc(sizeof(Complex) * template_width * template_height * 3);
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


	// Pad image signals
	Complex *h_padded_main_signal;
	Complex *h_padded_template_signal;

	int NEW_SIZE = PadData(h_main_signal, &h_padded_main_signal, main_signal_size, h_template_signal, &h_padded_template_signal, template_signal_size);

	cufftResult result;
	cufftHandle plan_input;
	cufftCreate(&plan_input);

	// cufftXtSetGPUs() - Define which GPUs to use
	int nGPUs = 2;
	int *whichGPUs;
	whichGPUs = (int*)malloc(sizeof(int) * nGPUs);

	// Iterate all device combinations to see if a supported combo exists
	for (int i = 0; i < GPU_N; i++)
	{
		for (int j = i + 1; j < GPU_N; j++)
		{
			whichGPUs[0] = i;
			whichGPUs[1] = j;
			result = cufftXtSetGPUs(plan_input, nGPUs, whichGPUs);

			if (result == CUFFT_INVALID_DEVICE) { continue; }
			else if (result == CUFFT_SUCCESS) { break; }
			else { printf("cufftXtSetGPUs failed\n"); exit(EXIT_FAILURE); }
		}

		if (result == CUFFT_SUCCESS) { break; }
	}

	if (result == CUFFT_INVALID_DEVICE)
	{
		printf("This sample requires two GPUs on the same board.\n");
		printf("No such board was found. Waiving sample.\n");
		return -1;
	}

	//Print the device information to run the code
	for (int i = 0; i < nGPUs; i++)
	{
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, whichGPUs[i]);
		printf("GPU Device %d: \"%s\" with compute capability %d.%d\n", whichGPUs[i], deviceProp.name, deviceProp.major, deviceProp.minor);

	}

	size_t* worksize;
	worksize = (size_t*)malloc(sizeof(size_t) * nGPUs);

	// cufftMakePlan1d() - Create the plan
	result = cufftMakePlan1d(plan_input, NEW_SIZE, CUFFT_C2C, 1, worksize);
	if (result != CUFFT_SUCCESS) { printf("*MakePlan* failed\n"); exit(EXIT_FAILURE); }

	// cufftMakePlan1d() - Create the plan
	result = cufftMakePlan1d(plan_input, NEW_SIZE, CUFFT_C2C, 1, worksize);
	if (result != CUFFT_SUCCESS) { printf("*MakePlan* failed\n"); exit(EXIT_FAILURE); }

	// cufftXtMalloc() - Malloc data on multiple GPUs
	cudaLibXtDesc *d_signal;
	result = cufftXtMalloc(plan_input, (cudaLibXtDesc **)&d_signal, CUFFT_XT_FORMAT_INPLACE);
	if (result != CUFFT_SUCCESS) { printf("*XtMalloc failed\n"); exit(EXIT_FAILURE); }
	cudaLibXtDesc *d_out_signal;
	result = cufftXtMalloc(plan_input, (cudaLibXtDesc **)&d_out_signal, CUFFT_XT_FORMAT_INPLACE);
	if (result != CUFFT_SUCCESS) { printf("*XtMalloc failed\n"); exit(EXIT_FAILURE); }
	cudaLibXtDesc *d_filter_kernel;
	result = cufftXtMalloc(plan_input, (cudaLibXtDesc **)&d_filter_kernel, CUFFT_XT_FORMAT_INPLACE);
	if (result != CUFFT_SUCCESS) { printf("*XtMalloc failed\n"); exit(EXIT_FAILURE); }
	cudaLibXtDesc *d_out_filter_kernel;
	result = cufftXtMalloc(plan_input, (cudaLibXtDesc **)&d_out_filter_kernel, CUFFT_XT_FORMAT_INPLACE);
	if (result != CUFFT_SUCCESS) { printf("*XtMalloc failed\n"); exit(EXIT_FAILURE); }

	// cufftXtMemcpy() - Copy data from host to multiple GPUs
	result = cufftXtMemcpy(plan_input, d_signal, h_padded_main_signal, CUFFT_COPY_HOST_TO_DEVICE);
	if (result != CUFFT_SUCCESS) { printf("*XtMemcpy failed\n"); exit(EXIT_FAILURE); }
	result = cufftXtMemcpy(plan_input, d_filter_kernel, h_padded_main_signal, CUFFT_COPY_HOST_TO_DEVICE);
	if (result != CUFFT_SUCCESS) { printf("*XtMemcpy failed\n"); exit(EXIT_FAILURE); }

	// cufftXtExecDescriptorC2C() - Execute FFT on data on multiple GPUs
	result = cufftXtExecDescriptorC2C(plan_input, d_signal, d_signal, CUFFT_FORWARD);
	if (result != CUFFT_SUCCESS) { printf("*XtExecC2C  failed\n"); exit(EXIT_FAILURE); }
	result = cufftXtExecDescriptorC2C(plan_input, d_filter_kernel, d_filter_kernel, CUFFT_FORWARD);
	if (result != CUFFT_SUCCESS) { printf("*XtExecC2C  failed\n"); exit(EXIT_FAILURE); }

	// cufftXtMemcpy() - Copy the data to natural order on GPUs
	result = cufftXtMemcpy(plan_input, d_out_signal, d_signal, CUFFT_COPY_DEVICE_TO_DEVICE);
	if (result != CUFFT_SUCCESS) { printf("*XtMemcpy failed\n"); exit(EXIT_FAILURE); }
	result = cufftXtMemcpy(plan_input, d_out_filter_kernel, d_filter_kernel, CUFFT_COPY_DEVICE_TO_DEVICE);
	if (result != CUFFT_SUCCESS) { printf("*XtMemcpy failed\n"); exit(EXIT_FAILURE); }

	printf("\n\nValue of Library Descriptor\n");
	printf("Number of GPUs %d\n", d_out_signal->descriptor->nGPUs);
	printf("Device id  %d %d\n", d_out_signal->descriptor->GPUs[0], d_out_signal->descriptor->GPUs[1]);
	printf("Data size on GPU %ld %ld\n", (long)(d_out_signal->descriptor->size[0] / sizeof(cufftComplex)), (long)(d_out_signal->descriptor->size[1] / sizeof(cufftComplex)));

	//Multiply the coefficients together and normalize the result
	printf("Launching ComplexPointwiseMulAndScale<<< >>>\n");
	multiplyCoefficient(d_out_signal, d_out_filter_kernel, NEW_SIZE, 1.0f / NEW_SIZE, nGPUs);

	// cufftXtExecDescriptorC2C() - Execute inverse  FFT on data on multiple GPUs
	printf("Transforming signal back cufftExecC2C\n");
	result = cufftXtExecDescriptorC2C(plan_input, d_out_signal, d_out_signal, CUFFT_INVERSE);
	if (result != CUFFT_SUCCESS) { printf("*XtExecC2C  failed\n"); exit(EXIT_FAILURE); }

	// Create host pointer pointing to padded signal
	Complex *h_convolved_signal = h_padded_main_signal;

	// Allocate host memory for the convolution result
	Complex *h_convolved_signal_ref = (Complex *)malloc(sizeof(Complex) * main_signal_size);

	// cufftXtMemcpy() - Copy data from multiple GPUs to host
	result = cufftXtMemcpy(plan_input, h_convolved_signal, d_out_signal, CUFFT_COPY_DEVICE_TO_HOST);
	if (result != CUFFT_SUCCESS) { printf("*XtMemcpy failed\n"); exit(EXIT_FAILURE); }

	// Convolve on the host
	Convolve(h_main_signal, main_signal_size, h_template_signal,
		template_signal_size, h_convolved_signal_ref);

	// Compare CPU and GPU result
	bool bTestResult = sdkCompareL2fe((float *)h_convolved_signal_ref,
		(float *)h_convolved_signal, 2 * main_signal_size,
		1e-5f);
	printf("\nvalue of TestResult %d\n", bTestResult);

	// Cleanup memory
	free(whichGPUs);
	free(worksize);
	free(h_main_signal);
	free(h_template_signal);
	free(h_padded_main_signal);
	free(h_padded_template_signal);
	free(h_convolved_signal_ref);

	// cudaXtFree() - Free GPU memory
	result = cufftXtFree(d_signal);
	if (result != CUFFT_SUCCESS) { printf("*XtFree failed\n"); exit(EXIT_FAILURE); }
	result = cufftXtFree(d_filter_kernel);
	if (result != CUFFT_SUCCESS) { printf("*XtFree failed\n"); exit(EXIT_FAILURE); }
	result = cufftXtFree(d_out_signal);
	if (result != CUFFT_SUCCESS) { printf("*XtFree failed\n"); exit(EXIT_FAILURE); }
	result = cufftXtFree(d_out_filter_kernel);
	if (result != CUFFT_SUCCESS) { printf("*XtFree failed\n"); exit(EXIT_FAILURE); }

	// cufftDestroy() - Destroy FFT plan
	result = cufftDestroy(plan_input);
	if (result != CUFFT_SUCCESS) { printf("cufftDestroy failed: code %d\n", (int)result); exit(EXIT_FAILURE); }

	// cudaDeviceReset causes the driver to clean up all state. While
	// not mandatory in normal operation, it is good practice.  It is also
	// needed to ensure correct operation when the application is being
	// profiled. Calling cudaDeviceReset causes all profile data to be
	// flushed before the application exitsits
	cudaDeviceReset();
	exit(bTestResult ? EXIT_SUCCESS : EXIT_FAILURE);
}

///////////////////////////////////////////////////////////////////////////////////
// Function for padding original data
//////////////////////////////////////////////////////////////////////////////////
int PadData(const Complex *signal, Complex **padded_signal, int signal_size,
	const Complex *filter_kernel, Complex **padded_filter_kernel, int filter_kernel_size)
{
	int minRadius = filter_kernel_size / 2;
	int maxRadius = filter_kernel_size - minRadius;
	int new_size = signal_size + maxRadius;

	// Pad signal
	Complex *new_data = (Complex *)malloc(sizeof(Complex) * new_size);
	memcpy(new_data + 0, signal, signal_size * sizeof(Complex));
	memset(new_data + signal_size, 0, (new_size - signal_size) * sizeof(Complex));
	*padded_signal = new_data;

	// Pad filter
	new_data = (Complex *)malloc(sizeof(Complex) * new_size);
	memcpy(new_data + 0, filter_kernel + minRadius, maxRadius * sizeof(Complex));
	memset(new_data + maxRadius, 0, (new_size - filter_kernel_size) * sizeof(Complex));
	memcpy(new_data + new_size - minRadius, filter_kernel, minRadius * sizeof(Complex));
	*padded_filter_kernel = new_data;

	return new_size;
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

////////////////////////////////////////////////////////////////////////////////
//  Launch Kernel on multiple GPU
////////////////////////////////////////////////////////////////////////////////
void  multiplyCoefficient(cudaLibXtDesc *d_signal, cudaLibXtDesc *d_filter_kernel,
	int new_size, float val, int nGPUs)
{
	int device;
	//Launch the ComplexPointwiseMulAndScale<<< >>> kernel on multiple GPU
	for (int i = 0; i < nGPUs; i++)
	{
		device = d_signal->descriptor->GPUs[i];

		//Set device
		cudaSetDevice(device);

		//Perform GPU computations
		ComplexPointwiseMulAndScale << <32, 256 >> >((cufftComplex*)d_signal->descriptor->data[i],
			(cufftComplex*)d_filter_kernel->descriptor->data[i],
			int(d_signal->descriptor->size[i] / sizeof(cufftComplex)), val);
	}

	// Wait for device to finish all operation
	for (int i = 0; i< nGPUs; i++)
	{
		device = d_signal->descriptor->GPUs[i];
		cudaSetDevice(device);
		cudaDeviceSynchronize();
	}
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

