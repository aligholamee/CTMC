#include <iostream>
#include <cuda_runtime.h>
#include <bitmap_image.hpp>
#include <ntm.cuh>

using namespace std;

int main(int argc, char* argv[]) {

	// Parse args
	for (int count = 0; count < argc; count += 2) {
		
		bitmap_image main_image(argv[count]);
		bitmap_image template_image(argv[count + 1]);

		// Initiate template matching
		fast_naive_template_match(main_image, template_image);
	}

	return 0;
}