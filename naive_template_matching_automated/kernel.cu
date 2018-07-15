#include <iostream>
#include <cuda_runtime.h>
#include <kernel.cuh>
#include <math_functions.h>

__global__ void compute_sad_array_kernel(
                                            int* sad_array, unsigned char* image,
                                            unsigned char* kernel, int sad_array_size,
                                            int image_width, int image_height,
                                            int kernel_width, int kernel_height,
                                            int kernel_size)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int SAD_RESULT = 0;

	if (row < image_height && col < image_width) {
		#pragma unroll 4
		for (int v_row = 0; v_row < kernel_height; v_row++) {
			#pragma unroll 4
			for (int v_col = 0; v_col < kernel_width; v_col++) {
				int m_r = (int)(image[((row + v_row) * image_width + (col + v_col)) * 3 + 0]);
				int m_g = (int)(image[((row + v_row) * image_width + (col + v_col)) * 3 + 1]);
				int m_b = (int)(image[((row + v_row) * image_width + (col + v_col)) * 3 + 2]);
				int t_r = (int)(kernel[(v_row * kernel_width + v_col) * 3 + 0]);
				int t_g = (int)(kernel[(v_row * kernel_width + v_col) * 3 + 1]);
				int t_b = (int)(kernel[(v_row * kernel_width + v_col) * 3 + 2]);
				int error = abs(m_r - t_r) + abs(m_g - t_g) + abs(m_b - t_b);
				SAD_RESULT += error;
			}
		}


		int NORMALIZED_SAD = (int)(SAD_RESULT / (float)kernel_size);

		__syncthreads();

		int my_index_in_sad_array = row * image_width + col;
		if (my_index_in_sad_array < sad_array_size) {
			sad_array[my_index_in_sad_array] = NORMALIZED_SAD;
		}
	}
}


__global__ void find_min_in_sad_array(
                                        int* sad_array, int sad_array_size,
                                        int* min_sad, int* mutex)
{
    unsigned int tId = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int stride = gridDim.x * blockDim.x;
	unsigned int offset = 0;

	__shared__ int cache[BLOCK_SIZE];

	int temp = 5000;
	while (tId + offset < sad_array_size) {
		temp = fminf(temp, sad_array[tId + offset]);
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

		*min_sad = fminf(*min_sad, cache[0]);

		// Unlock
		atomicExch(mutex, 0);
	}
}

__global__ void get_num_of_occurrences(
                                        int* sad_array, int* min_sad,
                                        int* mutex, int* num_occurrences
)
{
	unsigned int tId = threadIdx.x + blockIdx.x * blockDim.x;

	__shared__ int cache[BLOCK_SIZE];

	cache[threadIdx.x] = sad_array[tId];

	if (threadIdx.x == 0)
		cache[0] = 0;

	__syncthreads();

	if (cache[threadIdx.x] == *min_sad)
		atomicAdd(&cache[0], 1);

	__syncthreads();
    
	// Update global occurance for each block
	if (threadIdx.x == 0) {

		// Lock
		while (atomicCAS(mutex, 0, 1) != 0);

		atomicAdd(num_occurrences, cache[0]);

		// Unlock
		atomicExch(mutex, 0);
	}
}