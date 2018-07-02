#include <iostream>
#include <cuda_runtime.h>


#define funcCheck(stmt) do { cudaError_t err = stmt; if (err != cudaSuccess) { printf("[ERROR] Failed to run stmt %d, error body: %s\n", __LINE__, cudaGetErrorString(err)); return -1; } } while (0)

using namespace std;
 

int main(int argc,		// Number of arguments in array argv 
	char *argv[])	// Array of command-line argument strings
{
	// Display each command-line argument
	cout << "\nCommand-line Arguments: " << endl;

	for (int count = 0; count < argc; count++)
		cout << "argv["<< count <<"]: " << argv[count] << endl;

	for (int count = 0; count < argc; count++) {

		// Get BMP file
		// file
		// Extract main image matrix
		// int *h_imageMat = extractMatrix(file)

		// Extract template matrix
		// int *h_templateMat = extractMatrix(file)


		// Call match function

		// templateMatchSetup(h_imageMat, h_templateMat);

	}

	system("pause");
	return 0;
}

int templateMatchSetup(int *h_imageMat, int *h_templateMat)
{

	// Define device pointers
	int *d_imageMat;
	int *d_templateMat;

	// Get matrix sizes
	// imageMat Size
	// templateMat Size

	// Allocate space on device
	funcCheck(cudaMalloc((void**)&d_imageMat, imageMatSize));
	funcCheck(cudaMalloc((void**)&d_templateMat, templateMatSize));

	// Copy matrices to device
	funcCheck(cudaMemcpy(h_imageMat, d_imageMat, imageMatSize, cudaMemcpyHostToDevice));
	funcCheck(cudaMemcpy(h_templateMat, d_templateMat, templateMatSize, cudaMemcpyHostToDevice));

	




	// Release space on device
	funcCheck(cudaFree(d_imageMat));
	funcCheck(cudaFree(d_templateMat));

	return EXIT_SUCCESS;
}

