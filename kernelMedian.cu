#include <cuda.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <time.h>
#define TILE_SIZE 4
#define WINDOW_SIZE (3)


template<class IMG_TYPE>
__global__ void kernelMedian( const IMG_TYPE * __restrict__ in, IMG_TYPE *output, int j_dim, int pitch)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned char filterVector[9] = {0,0,0,0,0,0,0,0,0};
	if((row==0) || (col==0) || (row==pitch-1) || (col==j_dim-1))
		output[row*j_dim+col] = 0; //Deal with boundry conditions
	else {
		for (int x = 0; x < WINDOW_SIZE; x++) {
			for (int y = 0; y < WINDOW_SIZE; y++){
				filterVector[x*WINDOW_SIZE+y] = in[(row+x-1)*j_dim+(col+y-1)];   // setup the filterign window.
			}
		}
		for (int i = 0; i < 9; i++) {
			for (int j = i + 1; j < 9; j++) {
				if (filterVector[i] > filterVector[j]) {
					//Swap the variables.
					char tmp = filterVector[i];
					filterVector[i] = filterVector[j];
					filterVector[j] = tmp;
				}
			}
		}
		output[row*j_dim+col] = filterVector[4];   //Set the output variables.
	}
 }



 
