#include <stdio.h>

#define  N  10000

__global__ void productoEscalarAtomic(float *vector_a, float *vector_b, float *res){

__shared__ float cache[threadSize];
int iter = blockDim.x;
const int index = threadIdx.x + blockIdx.x*blockDim.x;

cache[threadIdx.x] = vector_a[index]*vector_b[index];
__syncthreads();

while(iter > 1){

if(threadIdx.x < iter/2){
cache[threadIdx.x] = cache[threadIdx.x] + cache[threadIdx.x + iter/2];
}

iter = iter/2;
__syncthreads();
}
if(threadIdx.x == 0)
atomicAdd(&res[0], cache[0]);
}

__global__ void productoEscalarNoAtomic(float *vector_a, float *vector_b, float *res){

__shared__ float cache[threadSize];
int iter = blockDim.x;
const int index = threadIdx.x + blockIdx.x*blockDim.x;

cache[threadIdx.x] = vector_a[index]*vector_b[index];
__syncthreads();

while(iter > 1){

if(threadIdx.x < iter/2){
cache[threadIdx.x] = cache[threadIdx.x] + cache[threadIdx.x + iter/2];
}

iter = iter/2;
__syncthreads();
}
if(threadIdx.x == 0)
res[0]= res[0]+cache[0]);
}

__global__ void productoEscalarAtomicAll(float *vector_a, float *vector_b, float *res){
__shared__ float cache[threadSize];
const int index = threadIdx.x + blockIdx.x*blockDim.x;

atomicAdd(&res[0], vector_a[index]*vector_b[index]);

}



int main(void){



}
