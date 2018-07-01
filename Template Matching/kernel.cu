#include <iostream>

using namespace std;

int main(int argc,		// Number of arguments in array argv 
	char *argv[])	// Array of command-line argument strings
{
	// Display each command-line argument
	cout << "\nCommand-line Arguments: " << endl;

	for (int count = 0; count < argc; count++)
		cout << "argv["<< count <<"]: " << argv[count] << endl;

	system("pause");
	return 0;
}