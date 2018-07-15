#ifndef NTM_H
#define NTM_H

#define M_PI 3.14159265
#define BLOCK_SIZE 1024
#define BLOCK_SIZE_X 32
#define BLOCK_SIZE_Y 32

/*
* Preprocessing of fast naive template matching
*/
int fast_naive_template_match(bitmap_image main_image, bitmap_image template_image);

/*
* Naive serial template matching
*/
void naive_template_match(bitmap_image main_image, bitmap_image template_image);

/*
* Serially get the number of occurances
*/

int get_num_of_occurrences(int * arr, unsigned int size);

/*
* Naive 90 degree anti-clockwise rotation
*/
bitmap_image rotate_anticw(bitmap_image image, unsigned int width, unsigned int height);


#endif