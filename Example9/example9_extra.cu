#include <stdio.h>

__global__ void hiloFinal(int *hilo){
int nonsense = 0;
for(long int i = 0; i<255-threadIdx.x;i++){
nonsense++;
}
atomicExch(&(hilo[0]),nonsense);
}

int main(void){

int *gpu_hilo;
int hilo;

cudaMalloc((void**) &gpu_hilo, sizeof(int));

hiloFinal<<<1,256>>>(gpu_hilo);

cudaMemcpy(&hilo, &(gpu_hilo[0]), sizeof(int), cudaMemcpyDeviceToHost);
printf("Hilo final: %d\n", hilo);

return 0;
}
