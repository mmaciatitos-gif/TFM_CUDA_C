#include <stdio.h>
#include <stdint.h>

typedef uint8_t sm;
__device__ cudaTextureObject_t tex;

__global__ void conway(sm *output, sm *liveArray, sm *breedArray){
int sum;
int x = threadIdx.x+blockDim.x*blockIdx.x;
int y = threadIdx.y+blockDim.y*blockIdx.y;
int edge = 0;

if(x == 0) edge = edge + 1;
if(y == 0) edge = edge + 2;

conwaySum(&sum, x, y, edge);

if (tex2D<sm>(tex, x, y) == 0 && breedArray[sum] == 1) output[x + y * blockDim.x * gridDim.x] = 1;
if (tex2D<sm>(tex, x, y) == 1 && liveArray[sum] == 0)  output[x + y * blockDim.x * gridDim.x] = 0;

}

__device__ int conwaySum(int *sum, int x, int y, int mode){
switch(mode){
case 0:
	sum =  
	tex2D<sm>(tex, (x - 1), 				y) +
	tex2D<sm>(tex, (x + 1) % blockDim.x*gridDim.x , 	y) +
	tex2D<sm>(tex, x, 					(y - 1)) +
	tex2D<sm>(tex, (x - 1), 				(y - 1)) +
	tex2D<sm>(tex, (x + 1) % blockDim.x*gridDim.x , 	(y - 1)) +
	tex2D<sm>(tex, x,				 	(y + 1) % blockDim.y*gridDim.y) +
	tex2D<sm>(tex, (x - 1),				 	(y + 1) % blockDim.y*gridDim.y) +
	tex2D<sm>(tex, (x + 1) % blockDim.x*gridDim.x , 	(y + 1) % blockDim.y*gridDim.y);
break;

case 1:
	sum =  
	tex2D<sm>(tex, blockDim.x*gridDim.x - 1,		y) +
	tex2D<sm>(tex, (x + 1) % blockDim.x*gridDim.x , 	y) +
	tex2D<sm>(tex, x, 					(y - 1)) +
	tex2D<sm>(tex, blockDim.x*gridDim.x - 1, 		(y - 1)) +
	tex2D<sm>(tex, (x + 1) % blockDim.x*gridDim.x , 	(y - 1)) +
	tex2D<sm>(tex, x,				 	(y + 1) % blockDim.y*gridDim.y) +
	tex2D<sm>(tex, blockDim.x*gridDim.x - 1,		(y + 1) % blockDim.y*gridDim.y) +
	tex2D<sm>(tex, (x + 1) % blockDim.x*gridDim.x , 	(y + 1) % blockDim.y*gridDim.y);
break;

case 2:
	sum =  
	tex2D<sm>(tex, blockDim.x*gridDim.x - 1,		y) +
	tex2D<sm>(tex, (x + 1) % blockDim.x*gridDim.x , 	y) +
	tex2D<sm>(tex, x, 					blockDim.y*gridDim.y - 1) +
	tex2D<sm>(tex, blockDim.x*gridDim.x - 1, 		blockDim.y*gridDim.y - 1) +
	tex2D<sm>(tex, (x + 1) % blockDim.x*gridDim.x , 	blockDim.y*gridDim.y - 1) +
	tex2D<sm>(tex, x,				 	(y + 1) % blockDim.y*gridDim.y) +
	tex2D<sm>(tex, blockDim.x*gridDim.x - 1,		(y + 1) % blockDim.y*gridDim.y) +
	tex2D<sm>(tex, (x + 1) % blockDim.x*gridDim.x , 	(y + 1) % blockDim.y*gridDim.y);
break;

case 3:
	sum =  
	tex2D<sm>(tex, blockDim.x*gridDim.x - 1,		y) +
	tex2D<sm>(tex, (x + 1) % blockDim.x*gridDim.x , 	y) +
	tex2D<sm>(tex, x, 					blockDim.y*gridDim.y - 1) +
	tex2D<sm>(tex, blockDim.x*gridDim.x - 1, 		blockDim.y*gridDim.y - 1)) +
	tex2D<sm>(tex, (x + 1) % blockDim.x*gridDim.x , 	blockDim.y*gridDim.y - 1)) +
	tex2D<sm>(tex, x,				 	(y + 1) % blockDim.y*gridDim.y) +
	tex2D<sm>(tex, blockDim.x*gridDim.x - 1,		(y + 1) % blockDim.y*gridDim.y) +
	tex2D<sm>(tex, (x + 1) % blockDim.x*gridDim.x , 	(y + 1) % blockDim.y*gridDim.y);
break;

}


}


}


int main(void){






return 0;
}
