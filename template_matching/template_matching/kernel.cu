
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>
#include <iostream>

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

struct BITMAP {
	int width;
	int height;
	unsigned char header[54];
	unsigned char *pixels;
	int size;
};

int initiate_parallel_template_matching(BITMAP, BITMAP);
void serial_template_matching(BITMAP, BITMAP);
BITMAP read_bitmap_image(string);
BITMAP rotate_bitmap_image(BITMAP, double);
void save_bitmap_image(string, BITMAP);
void device_query();
int find_minimum_in_array_in_serial(int*, unsigned int);
int get_num_of_occurances_in_serial(int*, unsigned int, int, bool);

/*
*	CUDA Kernel to compute MSEs
*/
__global__ void
computeMSEKernel(int* kernelMSEs, unsigned char* image, unsigned char* kernel, int kernelMSESize, int image_width, int image_height, int kernel_width, int kernel_height)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = 1;
	int virtualKernelMSE = 0;

	int virtual_kernel_row_start = row;
	int virtual_kernel_row_end = virtual_kernel_row_start + kernel_height;
	int virtual_kernel_col_start = col * stride;
	int virtual_kernel_col_end = virtual_kernel_row_start + kernel_width;

	if (virtual_kernel_col_end < image_width && virtual_kernel_row_end < image_height) {
		for (int kernelRow = 0; kernelRow < kernel_height; kernelRow++) {
			for (int kernelCol = 0; kernelCol < kernel_width; kernelCol++) {
				int imageRow = virtual_kernel_row_start + kernelRow;
				int imageCol = virtual_kernel_col_start + kernelCol;
				int m_r = int(image[(imageRow * image_width + imageCol) * 3] - '0');
				int m_g = int(image[(imageRow * image_width + imageCol) * 3 + 1] - '0');
				int m_b = int(image[(imageRow * image_width + imageCol) * 3 + 2] - '0');
				int t_r = int(kernel[(kernelRow * kernel_width + kernelCol) * 3] - '0');
				int t_g = int(kernel[(kernelRow * kernel_width + kernelCol) * 3 + 1] - '0');
				int t_b = int(kernel[(kernelRow * kernel_width + kernelCol) * 3 + 2] - '0');
				int error = (m_r - t_r) + (m_g - t_g) + (m_b - t_b);
				virtualKernelMSE += error;
			}
		}

		__syncthreads();

		int myIndexInKernelMSEsArray = row * image_width + col;
		if (myIndexInKernelMSEsArray < kernelMSESize) {
			kernelMSEs[myIndexInKernelMSEsArray] = virtualKernelMSE;
		}
	}
}


/*
*	CUDA Kernel to compute the minimum number in an array
*/
__global__ void
findMinInArrayKernel(int* kernelMSEs, int kernelMSESize, int* min_MSE, int* mutex)
{
	unsigned int tId = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int stride = gridDim.x * blockDim.x;
	unsigned int offset = 0;

	__shared__ int cache[BLOCK_SIZE];

	int temp = 1000;
	while (tId + offset < kernelMSESize) {
		temp = fminf(temp, kernelMSEs[tId + offset]);
		offset += stride;
	}

	cache[threadIdx.x] = temp;

	__syncthreads();

	unsigned int i = blockDim.x / 2;
	while (i != 0) {
		if (threadIdx.x < i) {
			cache[threadIdx.x] = fminf(cache[threadIdx.x], cache[threadIdx.x + i]);
		}

		__syncthreads();
		i /= 2;
	}

	// Update global min for each block
	if (threadIdx.x == 0) {

		// Lock
		while (atomicCAS(mutex, 0, 1) != 0);

		*min_MSE = fminf(*min_MSE, cache[0]);

		// Unlock
		atomicExch(mutex, 0);
	}
}
 
__global__ void 
findNumberofOccurances(int* kernelMSEs, int* min_MSE, int* mutex, int* num_occurances)
{
	unsigned int tId = threadIdx.x + blockIdx.x * blockDim.x;

	__shared__ int cache[BLOCK_SIZE];

	cache[threadIdx.x] = kernelMSEs[tId];
	
	if (threadIdx.x == 0)
		cache[0] = 0;

	__syncthreads();

	if (cache[threadIdx.x] == *min_MSE)
		atomicAdd(&cache[0], 1);


	__syncthreads();
	// Update global occurance for each block
	if (threadIdx.x == 0) {

		// Lock
		while (atomicCAS(mutex, 0, 1) != 0);

		atomicAdd(num_occurances, cache[0]);

		// Unlock
		atomicExch(mutex, 0);
	}
}
int main()
{
	BITMAP mainImage = read_bitmap_image("coin_col.bmp");
	BITMAP templateImage = read_bitmap_image("coin.bmp");

	// templateImage = rotate_bitmap_image(templateImage, 270);
	initiate_parallel_template_matching(mainImage, templateImage);
	wcout << "\n========================================" << endl;
	serial_template_matching(mainImage, templateImage);
	// device_query();
	system("pause");
	return 0;
}

int	initiate_parallel_template_matching(BITMAP mainImage, BITMAP templateImage)
{
	unsigned char* d_MainImage;
	unsigned char* d_TemplateImage;
	int height_difference = mainImage.height - templateImage.height;
	int width_difference = mainImage.width - templateImage.width;
	int kernel_MSE_size = (height_difference + 1) * (width_difference + 1);
	int* d_KernelMSEs;
	int* h_KernelMSEs;
	int* d_Min_MSE;
	int* h_Min_MSE;
	int* d_mutex;
	int* d_num_occurances;
	int* h_num_occurances;
	cudaEvent_t start;
	cudaEvent_t stop;
	float elapsed_time = 0.0f;

	errorHandler(cudaMalloc((void **)&d_MainImage, mainImage.size * sizeof(unsigned char)));
	errorHandler(cudaMalloc((void **)&d_TemplateImage, templateImage.size * sizeof(unsigned char)));
	errorHandler(cudaMalloc((void **)&d_KernelMSEs, kernel_MSE_size * sizeof(int)));
	errorHandler(cudaMalloc((void **)&d_Min_MSE, sizeof(int)));
	errorHandler(cudaMalloc((void **)&d_mutex, sizeof(int)));
	errorHandler(cudaMalloc((void **)&d_num_occurances, sizeof(int)));
	errorHandler(cudaMemset(d_Min_MSE, 0, sizeof(int)));
	errorHandler(cudaMemset(d_mutex, 0, sizeof(int)));
	errorHandler(cudaMemset(d_num_occurances, 0, sizeof(int)));
	h_KernelMSEs = new int[kernel_MSE_size];
	h_Min_MSE = new int[1];
	h_num_occurances = new int[1];
	errorHandler(cudaMemcpy(d_MainImage, mainImage.pixels, mainImage.size * sizeof(unsigned char), cudaMemcpyHostToDevice));
	errorHandler(cudaMemcpy(d_TemplateImage, templateImage.pixels, templateImage.size * sizeof(unsigned char), cudaMemcpyHostToDevice));
	errorHandler(cudaEventCreate(&start));
	errorHandler(cudaEventCreate(&stop));
	errorHandler(cudaEventRecord(start));

	dim3 grid_dimensions(ceil((float)mainImage.width / BLOCK_SIZE_X), ceil((float)mainImage.height / BLOCK_SIZE_Y), 1);
	dim3 block_dimensions(BLOCK_SIZE_X, BLOCK_SIZE_Y, 1);
	computeMSEKernel << <grid_dimensions, block_dimensions >> > (d_KernelMSEs, d_MainImage, d_TemplateImage, kernel_MSE_size, mainImage.width, mainImage.height, templateImage.width, templateImage.height);

	dim3 grid_dimensions_2(ceil((float)kernel_MSE_size) / BLOCK_SIZE, 1, 1);
	dim3 block_dimensions_2(BLOCK_SIZE, 1, 1);
	findMinInArrayKernel << <grid_dimensions_2, block_dimensions_2 >> > (d_KernelMSEs, kernel_MSE_size, d_Min_MSE, d_mutex);

	findNumberofOccurances << < grid_dimensions_2, block_dimensions_2 >> > (d_KernelMSEs, d_Min_MSE, d_mutex, d_num_occurances);
	errorHandler(cudaGetLastError());
	errorHandler(cudaEventRecord(stop, NULL));
	errorHandler(cudaEventSynchronize(stop));
	errorHandler(cudaEventElapsedTime(&elapsed_time, start, stop));
	errorHandler(cudaMemcpy(h_KernelMSEs, d_KernelMSEs, kernel_MSE_size * sizeof(int), cudaMemcpyDeviceToHost));
	errorHandler(cudaMemcpy(h_Min_MSE, d_Min_MSE, sizeof(int), cudaMemcpyDeviceToHost));
	errorHandler(cudaMemcpy(h_num_occurances, d_num_occurances, sizeof(int), cudaMemcpyDeviceToHost));

	wcout << "[[[ Parallel Computation Results ]]] " << endl;
	wcout << "Elapsed time in msec = " << elapsed_time << endl;
	wcout << "[Main Image Dimensions]: " << mainImage.height << "*" << mainImage.width << endl;
	wcout << "[Template Image Dimensions]: " << templateImage.height << "*" << templateImage.width << endl;
	wcout << "[MSE Array Size]:	" << kernel_MSE_size << endl;
	wcout << "[Found Minimum]:  " << *h_Min_MSE << endl;
	wcout << "[Number of occurances]: " << *h_num_occurances;
	errorHandler(cudaFree(d_MainImage));
	errorHandler(cudaFree(d_TemplateImage));

	return EXIT_SUCCESS;
}

void serial_template_matching(BITMAP mainImage, BITMAP templateImage)
{
	int height_difference = mainImage.height - templateImage.height;
	int width_difference = mainImage.width - templateImage.width;
	int MSE_size = (height_difference + 1) * (width_difference + 1);
	int * mseArray = new int[MSE_size];

	for (int row = 0; row < mainImage.height; row++) {
		for (int col = 0; col < mainImage.width; col++) {
			if (row + templateImage.height < mainImage.height && col + templateImage.width < mainImage.width) {
				int indexInsideMSEArray = row * mainImage.width + col;
				if (indexInsideMSEArray < MSE_size) {
					for (int i = 0; i < templateImage.height; i++) {
						for (int j = 0; j < templateImage.width; j++) {
							int vRow = row + i;
							int vCol = col + j;
							int m_r = int(mainImage.pixels[(vRow * mainImage.width + vCol) * 3] - '0');
							int m_g = int(mainImage.pixels[(vRow * mainImage.width + vCol) * 3 + 1] - '0');
							int m_b = int(mainImage.pixels[(vRow * mainImage.width + vCol) * 3 + 2] - '0');
							int t_r = int(templateImage.pixels[(i * templateImage.width + j) * 3] - '0');
							int t_g = int(templateImage.pixels[(i * templateImage.width + j) * 3 + 1] - '0');
							int t_b = int(templateImage.pixels[(i * templateImage.width + j) * 3 + 2] - '0');
							int error = (m_r - t_r) + (m_g - t_g) + (m_b - t_b);
							mseArray[indexInsideMSEArray] += error;
						}
					}
				}
			}
		}
	}
	wcout << "[[[ Serial Computation Results ]]] " << endl;
	wcout << "[Main Image Dimensions]: " << mainImage.height << "*" << mainImage.width << endl;
	wcout << "[Template Image Dimensions]: " << templateImage.height << "*" << templateImage.width << endl;
	wcout << "[MSE Array Size]:	" << MSE_size << endl;
	wcout << "[Number of occurances]: " << get_num_of_occurances_in_serial(mseArray, MSE_size, 0, true) << endl;
}

BITMAP read_bitmap_image(string file_name)
{
	BITMAP image;
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

	fwrite(image.pixels, sizeof(unsigned char), image.size, out);
	fclose(out);
}

void device_query()
{
	const int kb = 1024;
	const int mb = kb * kb;
	wcout << "NBody.GPU" << endl << "=========" << endl << endl;

	wcout << "CUDA version:   v" << CUDART_VERSION << endl;

	int devCount;
	cudaGetDeviceCount(&devCount);
	wcout << "CUDA Devices: " << endl << endl;

	for (int i = 0; i < devCount; ++i)
	{
		cudaDeviceProp props;
		cudaGetDeviceProperties(&props, i);
		wcout << i << ": " << props.name << ": " << props.major << "." << props.minor << endl;
		wcout << "  Global memory:   " << props.totalGlobalMem / mb << "mb" << endl;
		wcout << "  Shared memory:   " << props.sharedMemPerBlock / kb << "kb" << endl;
		wcout << "  Constant memory: " << props.totalConstMem / kb << "kb" << endl;
		wcout << "  Block registers: " << props.regsPerBlock << endl << endl;

		wcout << "  Warp size:         " << props.warpSize << endl;
		wcout << "  Threads per block: " << props.maxThreadsPerBlock << endl;
		wcout << "  Max block dimensions: [ " << props.maxThreadsDim[0] << ", " << props.maxThreadsDim[1] << ", " << props.maxThreadsDim[2] << " ]" << endl;
		wcout << "  Max grid dimensions:  [ " << props.maxGridSize[0] << ", " << props.maxGridSize[1] << ", " << props.maxGridSize[2] << " ]" << endl;
		wcout << "  Concurrent Kernels:		" << props.concurrentKernels;

		wcout << endl;
	}
}

int find_minimum_in_array_in_serial(int* arr, unsigned int arr_size)
{
	int minimum = 1000;
	for (int i = 0; i < arr_size; i++) {
		if (arr[i] < minimum) {
			minimum = arr[i];
		}
	}
	wcout << "[Found Minimum]:  " << minimum << endl;
	return minimum;
}

int get_num_of_occurances_in_serial(int* arr, unsigned int arr_size, int min_value, bool find_min_value)
{
	int occurance;
	int num_of_occurances = 0;

	if (!find_min_value) {
		occurance = min_value;
	}
	else {
		occurance = find_minimum_in_array_in_serial(arr, arr_size);
	}

	for (int i = 0; i < arr_size; i++)
		if (arr[i] == occurance)
			num_of_occurances++;

	return num_of_occurances;
}