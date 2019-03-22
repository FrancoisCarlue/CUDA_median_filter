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



template<class IMG_TYPE> 
__global__ void medianFilterSharedKernel(const IMG_TYPE * __restrict__ inputImageKernel, IMG_TYPE *outputImagekernel, int imageWidth, int imageHeight)
{
	//Set the row and col value for each thread.
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ unsigned char sharedmem[(TILE_SIZE+2)]  [(TILE_SIZE+2)];  //initialize shared memory
	//Take some values.
	bool is_x_left = (threadIdx.x == 0), is_x_right = (threadIdx.x == TILE_SIZE-1);
	bool is_y_top = (threadIdx.y == 0), is_y_bottom = (threadIdx.y == TILE_SIZE-1);

	//Initialize with zero
	if(is_x_left)
		sharedmem[threadIdx.x][threadIdx.y+1] = 0;
	else if(is_x_right)
		sharedmem[threadIdx.x + 2][threadIdx.y+1]=0;
	if (is_y_top){
		sharedmem[threadIdx.x+1][threadIdx.y] = 0;
		if(is_x_left)
			sharedmem[threadIdx.x][threadIdx.y] = 0;
		else if(is_x_right)
			sharedmem[threadIdx.x+2][threadIdx.y] = 0;
	}
	else if (is_y_bottom){
		sharedmem[threadIdx.x+1][threadIdx.y+2] = 0;
		if(is_x_right)
			sharedmem[threadIdx.x+2][threadIdx.y+2] = 0;
		else if(is_x_left)
			sharedmem[threadIdx.x][threadIdx.y+2] = 0;
	}

	//Setup pixel values
	sharedmem[threadIdx.x+1][threadIdx.y+1] = inputImageKernel[row*imageWidth+col];
	//Check for boundry conditions.
	if(is_x_left && (col>0))
		sharedmem[threadIdx.x][threadIdx.y+1] = inputImageKernel[row*imageWidth+(col-1)];
	else if(is_x_right && (col<imageWidth-1))
		sharedmem[threadIdx.x + 2][threadIdx.y+1]= inputImageKernel[row*imageWidth+(col+1)];
	if (is_y_top && (row>0)){
		sharedmem[threadIdx.x+1][threadIdx.y] = inputImageKernel[(row-1)*imageWidth+col];
		if(is_x_left)
			sharedmem[threadIdx.x][threadIdx.y] = inputImageKernel[(row-1)*imageWidth+(col-1)];
		else if(is_x_right )
			sharedmem[threadIdx.x+2][threadIdx.y] = inputImageKernel[(row-1)*imageWidth+(col+1)];
	}
	else if (is_y_bottom && (row<imageHeight-1)){
		sharedmem[threadIdx.x+1][threadIdx.y+2] = inputImageKernel[(row+1)*imageWidth + col];
		if(is_x_right)
			sharedmem[threadIdx.x+2][threadIdx.y+2] = inputImageKernel[(row+1)*imageWidth+(col+1)];
		else if(is_x_left)
			sharedmem[threadIdx.x][threadIdx.y+2] = inputImageKernel[(row+1)*imageWidth+(col-1)];
	}

	__syncthreads();   //Wait for all threads to be done.

	//Setup the filter.
	unsigned char filterVector[9] = {sharedmem[threadIdx.x][threadIdx.y], sharedmem[threadIdx.x+1][threadIdx.y], sharedmem[threadIdx.x+2][threadIdx.y],
																	 sharedmem[threadIdx.x][threadIdx.y+1], sharedmem[threadIdx.x+1][threadIdx.y+1], sharedmem[threadIdx.x+2][threadIdx.y+1],
																	 sharedmem[threadIdx.x] [threadIdx.y+2], sharedmem[threadIdx.x+1][threadIdx.y+2], sharedmem[threadIdx.x+2][threadIdx.y+2]};


	{
		for (int i = 0; i < 9; i++) {
			for (int j = i + 1; j < 9; j++) {
				if (filterVector[i] > filterVector[j]) {
					//Swap Values.
					char tmp = filterVector[i];
					filterVector[i] = filterVector[j];
					filterVector[j] = tmp;
				}
			}
		}
		outputImagekernel[row*imageWidth+col] = filterVector[4];   //Set the output image values.
	}
}

 
