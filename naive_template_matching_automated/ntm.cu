#include <iostream>
#include <chrono>
#include <bitmap_image.hpp>
#include <ntm.cuh>
#include <kernel.cuh>


using namespace std;

int fast_naive_template_match(bitmap_image main_image, bitmap_image template_image)
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

	// Define host pointers
	unsigned char* h_main_image;
	unsigned char* h_template_image;
	int* h_mse_array;
	int* h_min_mse;
	int* h_num_occurances;
	int* h_template_image_rotated;

	// Define device pointers
	unsigned char* d_main_image;
	unsigned char* d_template_image;
	int* d_mse_array;
	int* d_min_mse;
	int* d_num_occurances;
	int* d_mutex;

	// CUDA time handling
	cudaEvent_t start;
	cudaEvent_t stop;
	float elapsed_time = 0.0f;

	// Host allocation

	/*
	Extract Matrices
	*/

	h_main_image = new unsigned char[3 * main_size];

	for (int col = 0; col < main_width; col++) {
		for (int row = 0; row < main_height; row++) {
			rgb_t colors;

			main_image.get_pixel(col, row, colors);
			h_main_image[(row * main_width + col) * 3 + 0] = colors.red;
			h_main_image[(row * main_width + col) * 3 + 1] = colors.green;
			h_main_image[(row * main_width + col) * 3 + 2] = colors.blue;
		}
	}

	h_template_image = new unsigned char[3 * template_size];

	for (int col = 0; col < template_width; col++) {
		for (int row = 0; row < template_height; row++) {
			rgb_t colors;

			template_image.get_pixel(col, row, colors);
			h_template_image[(row * template_width + col) * 3 + 0] = colors.red;
			h_template_image[(row * template_width + col) * 3 + 1] = colors.green;
			h_template_image[(row * template_width + col) * 3 + 2] = colors.blue;
		}
	}

	/*
	*************************
	*/

	h_mse_array = new int[mse_array_size];
	h_min_mse = new int[1];
	h_num_occurances = new int[1];

	// Device allocation
	errorHandler(cudaMalloc((void **)&d_main_image, 3 * main_size * sizeof(unsigned char)));
	errorHandler(cudaMalloc((void **)&d_template_image, 3 * template_size * sizeof(unsigned char)));
	errorHandler(cudaMalloc((void **)&d_mse_array, mse_array_size * sizeof(int)));
	errorHandler(cudaMalloc((void **)&d_min_mse, sizeof(int)));
	errorHandler(cudaMalloc((void **)&d_mutex, sizeof(int)));
	errorHandler(cudaMalloc((void **)&d_num_occurances, sizeof(int)));
	errorHandler(cudaMemset(d_min_mse, 20, sizeof(int)));
	errorHandler(cudaMemset(d_mutex, 0, sizeof(int)));
	errorHandler(cudaMemset(d_num_occurances, 0, sizeof(int)));
	errorHandler(cudaMemcpy(d_main_image, h_main_image, 3 * main_size * sizeof(unsigned char), cudaMemcpyHostToDevice));
	errorHandler(cudaMemcpy(d_template_image, h_template_image, 3 * template_size * sizeof(unsigned char), cudaMemcpyHostToDevice));

	errorHandler(cudaEventCreate(&start));
	errorHandler(cudaEventCreate(&stop));
	errorHandler(cudaEventRecord(start));

	dim3 grid_dimensions((unsigned int)ceil((float)(main_width) / BLOCK_SIZE_X), (unsigned int)ceil((float)(main_height) / BLOCK_SIZE_Y), 1);
    dim3 block_dimensions(BLOCK_SIZE_X, BLOCK_SIZE_Y, 1);
    
    // kernel launch
	compute_sad_array_kernel << <grid_dimensions, block_dimensions >> > (d_mse_array, d_main_image, d_template_image, mse_array_size, main_width, main_height, template_width, template_height, template_size);

	dim3 grid_dimensions_2((unsigned int)ceil((float)mse_array_size) / BLOCK_SIZE, 1, 1);
    dim3 block_dimensions_2(BLOCK_SIZE, 1, 1);
    
    // Kernel launch
	find_min_in_sad_array_kernel << <grid_dimensions_2, block_dimensions_2 >> > (d_mse_array, mse_array_size, d_min_mse, d_mutex);

    // Kernel launch
    get_num_of_occurrences_kernel << < grid_dimensions_2, block_dimensions_2 >> > (d_mse_array, d_min_mse, d_mutex, d_num_occurances);
    
	errorHandler(cudaGetLastError());
	errorHandler(cudaEventRecord(stop, NULL));
	errorHandler(cudaEventSynchronize(stop));
	errorHandler(cudaEventElapsedTime(&elapsed_time, start, stop));
	errorHandler(cudaMemcpy(h_mse_array, d_mse_array, mse_array_size * sizeof(int), cudaMemcpyDeviceToHost));
	errorHandler(cudaMemcpy(h_min_mse, d_min_mse, sizeof(int), cudaMemcpyDeviceToHost));
	errorHandler(cudaMemcpy(h_num_occurances, d_num_occurances, sizeof(int), cudaMemcpyDeviceToHost));

	wcout << "[[[ Parallel Computation Results ]]] " << endl << endl;
	wcout << "[Elapsed time in msec]: " << (int)(elapsed_time) << endl;
	wcout << "[Main Image Dimensions]: " << main_width << "*" << main_height << endl;
	wcout << "[Template Image Dimensions]: " << template_width << "*" << template_height << endl;
	wcout << "[MSE Array Size]:	" << mse_array_size << endl;
	wcout << "[Found Minimum]:  " << *h_min_mse << endl;
	wcout << "[Number of Occurances]: " << *h_num_occurances << endl;
	errorHandler(cudaFree(d_main_image));
	errorHandler(cudaFree(d_template_image));
	free(h_main_image);
	free(h_template_image);
	return EXIT_SUCCESS;
}


void naive_template_match(bitmap_image main_image, bitmap_image template_image)
{
	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
	int main_width = main_image.width();
	int main_height = main_image.height();
	int template_width = template_image.width();
	int template_height = template_image.height();

	int templateSize = template_height * template_width;

	int THRESHOLD = 20;
	unsigned int NUM_OCCURRENCES = 0;
	int FOUND_MINIMUM = 100000;
	int NUM_OF_ZEROS = 0;

	wcout << "[[[ Initiated Serial Template Matching ]]] " << endl;

	for (int col = 0; col < main_width - template_width; col++) {
		for (int row = 0; row < main_height - template_height; row++) {

			int SUM_OF_ABSOLUTE_DEVIATIONS = 0;

			for (int j = 0; j < template_width; j++) {
				for (int i = 0; i < template_height; i++) {

					int mRow = row + i;
					int mCol = col + j;

					rgb_t m_color;
					rgb_t t_color;

					mainImage.get_pixel(mCol, mRow, m_color);
					templateImage.get_pixel(j, i, t_color);

					SUM_OF_ABSOLUTE_DEVIATIONS += abs(m_color.red - t_color.red)
												+ abs(m_color.green - t_color.green)
												+ abs(m_color.blue - t_color.blue);

				}
			}

			int NORMALIZED_SAD = (int)(SUM_OF_ABSOLUTE_DEVIATIONS / (float)templateSize);

			if (NORMALIZED_SAD < THRESHOLD) {
				NUM_OCCURRENCES++;
			}

			if (NORMALIZED_SAD < FOUND_MINIMUM) {
				FOUND_MINIMUM = (int)NORMALIZED_SAD;
			}

			if (NORMALIZED_SAD == 0)
				NUM_OF_ZEROS++;

		}
	}

	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	wcout << "[[[ Serial Computation Results ]]] " << endl << endl;
	wcout << "[Elapsed time in msec]: " << chrono::duration_cast<std::chrono::milliseconds> (end - begin).count() << endl;
	wcout << "[Main Image Dimensions]: " << main_width << "*" << main_height << endl;
	wcout << "[Template Image Dimensions]: " << template_width << "*" << template_height << endl;
	wcout << "[Found Minimum]:  " << FOUND_MINIMUM << endl;
	wcout << "[Number of Occurances]: " << NUM_OCCURRENCES << endl;
}


int get_num_of_occurrences(int * arr, unsigned int size)
{
	int min = arr[0];
	int num_of_occurs = 0;
	ofstream filemy;
	filemy.open("output.txt");

	for (unsigned int i = 0; i < size; i++) {
		filemy << arr[i] << "\n";
		if (arr[i] < min) {
			num_of_occurs = 1;
			min = arr[i];
		}

		if (arr[i] == min)
			num_of_occurs++;
	}

	wcout << "[Found Minimum]:  " << min << endl;
	wcout << "[Number of Occurances]: " << num_of_occurs << endl;

	return num_of_occurs;
}

bitmap_image rotate_anticw(bitmap_image image, unsigned int width, unsigned int height)
{
	bitmap_image result(height, width);
	for (unsigned int x = 0; x < width; x++) {
		for (unsigned int y = 0; y < (height + 1) / 2; y++) {
			rgb_t pixelColor;

			image.get_pixel(x, y, pixelColor);
			result.set_pixel(y, height - 1 - x, pixelColor.red, pixelColor.green, pixelColor.blue);

			image.get_pixel(width - 1 - y, x, pixelColor);
			result.set_pixel(x, y, pixelColor.red, pixelColor.green, pixelColor.blue);

			image.get_pixel(width - 1 - x, width - 1 - y, pixelColor);
			result.set_pixel(width - 1 - y, x, pixelColor.red, pixelColor.green, pixelColor.blue);

			image.get_pixel(y, width - 1 - x, pixelColor);
			result.set_pixel(width - 1 - x, height - 1 - y, pixelColor.red, pixelColor.green, pixelColor.blue);
		}
	}
	return result;
}