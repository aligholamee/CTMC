#ifndef KERNEL_CUH
#define KERNEL_CUH

/*
* Important launch parameters
*/
#define M_PI 3.14159265
#define BLOCK_SIZE 1024
#define BLOCK_SIZE_X 32
#define BLOCK_SIZE_Y 32

/*
* Handle CUDA errors
*/ 
#define errorHandler(stmt)																					\
	do {																									\
		cudaError_t err = stmt;																				\
		if (err != cudaSuccess) {                                                                           \
            std::wcout << "[ERROR] Failed to run statement" << __LINE__ << std::endl;                       \
            std::wcout << "Error body: " << cudaGetErrorString(err);										\
			return -1; }																 	 				\
	} while (0)																								\


/*
* Computes SAD array for all pixels of main image
*/
__global__ void compute_sad_array_kernel(
                                            int* sad_array, unsigned char* image,
                                            unsigned char* kernel, int sad_array_size,
                                            int image_width, int image_height,
                                            int kernel_width, int kernel_height,
                                            int kernel_size);


/*
* Efficient kernel to find the minimum in SAD array
*/
__global__ void find_min_in_sad_array_kernel(
                                                int* sad_array, int sad_array_size,
                                                int* min_sad, int* mutex);


/*
* Efficient kernel to count occurrences of minimum in SAD array
*/
__global__ void get_num_of_occurrences_kernel(
                                                int* sad_array, int* min_sad,
                                                int* mutex, int* num_occurrences
);

#endif