#include <iostream>
#include <cuda_runtime.h>

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
		// templateMatch(h_imageMat, h_templateMat);

	}

	system("pause");
	return 0;
}

int templateMatch(int *h_imageMat, int *h_templateMat)
{

	// Define device pointers
	int *d_imageMat;
	int *d_templateMat;


	// Allocate space on device





	return EXIT_SUCCESS;
}

