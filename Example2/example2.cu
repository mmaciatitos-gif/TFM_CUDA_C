#include <stdio.h>

__global__ void suma(int *a, int *b, int *c){

*c = *a + *b;
}

int main(){

int a, b, c;
int *gpu_a, *gpu_b, *gpu_c;

printf("Indica el valor de a: ");
scanf("%d", &a);
printf("Indica el valor de b: ");
scanf("%d", &b);


cudaMalloc((void**)&gpu_a, sizeof(int));
cudaMalloc((void**)&gpu_b, sizeof(int));
cudaMalloc((void**)&gpu_c, sizeof(int));

cudaMemcpy(gpu_a, &a, sizeof(int), cudaMemcpyHostToDevice);
cudaMemcpy(gpu_b, &b, sizeof(int), cudaMemcpyHostToDevice);

suma<<<1,1>>>(gpu_a, gpu_b, gpu_c);
cudaMemcpy(&c, gpu_c, sizeof(int), cudaMemcpyDeviceToHost);
printf("%d + %d = %d \n", a, b, c);

cudaFree(gpu_a);
cudaFree(gpu_b);
cudaFree(gpu_c);

return 0;
}
