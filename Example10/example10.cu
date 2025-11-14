#include <stdio.h>
#include <stdint.h>

#define N 100000000
#define partial_N 10000

__global__ void sumaVectorial(int *gpu_a, int *gpu_b, int *gpu_result, int size) {
  int index = threadIdx.x + blockDim.x*blockIdx.x;
  if(index < size) gpu_result[index] = gpu_a[index] + gpu_b[index];
}




int main(void){

//Definimos lo necesario para medir los tiempos de ejecución
//Creamos las variables evento
cudaEvent_t  start, stop;

//Creamos los eventos
cudaEventCreate (&start);
cudaEventCreate (&stop);

//Creamos una variable para almacenar los tiempos y otra para hacer media:
float timeTemp = 0;
float timeAvg;

//Definimos los vectores a sumar y la suma
int *vect_a, *vect_b, *vect_res;

vect_a = (int*) malloc(sizeof(int)*N);
vect_b = (int*) malloc(sizeof(int)*N);
vect_res = (int*) malloc(sizeof(int)*N);

for(int i = 0; i < N; i++){
vect_a[i] = i % 32;
vect_b[i] = i % 64;
}


//########################################################################################################################

//Implementacion base

//Definimos los vectores
int *gpu_a, *gpu_b, *gpu_res;
int size = partial_N *sizeof(int);

//Alocamos la memoria
cudaMalloc((void**)&gpu_a, size);
cudaMalloc((void**)&gpu_b, size);
cudaMalloc((void**)&gpu_res, size);

//Reseteamos el tiempo medio
timeAvg = 0;

//Creamos un bucle
for(int k=0; k<100; k++){

//Grabamos el evento inicial
cudaEventRecord(start,0);

//Comenzamos a sumar por partes
int i;
for(i = 0; i < N; i = i + partial_N){
cudaMemcpy(gpu_a, &vect_a[i], size, cudaMemcpyHostToDevice);
cudaMemcpy(gpu_b, &vect_b[i], size, cudaMemcpyHostToDevice);
sumaVectorial<<<partial_N/64+1, 64>>>(gpu_a, gpu_b, gpu_res, partial_N);
cudaMemcpy( &vect_res[i], gpu_res, size, cudaMemcpyDeviceToHost);
}

//Sumamos los elementos restantes
if((N-i) > 0){
cudaMemcpy(gpu_a, &vect_a[i], (N-i)*sizeof(int), cudaMemcpyHostToDevice);
cudaMemcpy(gpu_b, &vect_b[i], (N-i)*sizeof(int), cudaMemcpyHostToDevice);
sumaVectorial<<<(N-i)/64+1, 64>>>(gpu_a, gpu_b, gpu_res, (N-i));
cudaMemcpy( &vect_res[i], gpu_res, (N-i)*sizeof(int), cudaMemcpyDeviceToHost);
}

cudaDeviceSynchronize();

//Grabamos el evento final
cudaEventRecord(stop,0);
cudaEventSynchronize(stop);

//Obtenemos el tiempo de ejecucion
cudaEventElapsedTime(&timeTemp, start, stop);
timeAvg = timeAvg + timeTemp;
}


//Imprimimos la media de tiempo de ejecucion
printf("Tiempo original: %f\n", timeAvg/100.0);


cudaFree(gpu_a);
cudaFree(gpu_b);
cudaFree(gpu_res);

//########################################################################################################################

//Implementacion con streams



cudaStream_t s_1, s_2;
cudaStreamCreate(&s_1);
cudaStreamCreate(&s_2);

//Definimos los vectores
int *gpu_a_1, *gpu_b_1, *gpu_res_1;
int *gpu_a_2, *gpu_b_2, *gpu_res_2;

//Alocamos la memoria
cudaMalloc((void**)&gpu_a_1, size);
cudaMalloc((void**)&gpu_b_1, size);
cudaMalloc((void**)&gpu_res_1, size);
cudaMalloc((void**)&gpu_a_2, size);
cudaMalloc((void**)&gpu_b_2, size);
cudaMalloc((void**)&gpu_res_2, size);

int j;

//Reseteamos el tiempo medio
timeAvg = 0;

//Creamos un bucle
for(int k=0; k<100; k++){

//Grabamos el evento inicial
cudaEventRecord(start,0);

//Comenzamos a sumar por partes
j = 1;
int i;
for(i = 0; i < N; i = i + partial_N){
//Alternamos los streams
j = -j;
if(j == 1){
cudaMemcpyAsync(gpu_a_1, &vect_a[i], size, cudaMemcpyHostToDevice, s_1);
cudaMemcpyAsync(gpu_b_1, &vect_b[i], size, cudaMemcpyHostToDevice, s_1);
sumaVectorial<<<partial_N/64+1, 64, 0, s_1>>>(gpu_a_1, gpu_b_1, gpu_res_1, partial_N);
cudaMemcpyAsync( &vect_res[i], gpu_res_1, size, cudaMemcpyDeviceToHost, s_1);
}
else{
cudaMemcpyAsync(gpu_a_2, &vect_a[i], size, cudaMemcpyHostToDevice, s_2);
cudaMemcpyAsync(gpu_b_2, &vect_b[i], size, cudaMemcpyHostToDevice, s_2);
sumaVectorial<<<partial_N/64+1, 64, 0, s_2>>>(gpu_a_2, gpu_b_2, gpu_res_2, partial_N);
cudaMemcpyAsync( &vect_res[i], gpu_res_2, size, cudaMemcpyDeviceToHost, s_2);
}
}

//Sumamos los elementos restantes
if((N-i) > 0){
if(j == 1){
cudaMemcpyAsync(gpu_a_1, &vect_a[i], (N-i)*sizeof(int), cudaMemcpyHostToDevice, s_1);
cudaMemcpyAsync(gpu_b_1, &vect_b[i], (N-i)*sizeof(int), cudaMemcpyHostToDevice, s_1);
sumaVectorial<<<(N-i)/64+1, 64, 0, s_1>>>(gpu_a_1, gpu_b_1, gpu_res_1, (N-i));
cudaMemcpyAsync( &vect_res[i], gpu_res_1, (N-i)*sizeof(int), cudaMemcpyDeviceToHost, s_1);
}
else{
cudaMemcpyAsync(gpu_a_2, &vect_a[i], (N-i)*sizeof(int), cudaMemcpyHostToDevice, s_2);
cudaMemcpyAsync(gpu_b_2, &vect_b[i], (N-i)*sizeof(int), cudaMemcpyHostToDevice, s_2);
sumaVectorial<<<(N-i)/64+1, 64, 0, s_2>>>(gpu_a_2, gpu_b_2, gpu_res_2, (N-i));
cudaMemcpyAsync( &vect_res[i], gpu_res_2, (N-i)*sizeof(int), cudaMemcpyDeviceToHost, s_2);
}
}

cudaStreamSynchronize(s_1);
cudaStreamSynchronize(s_2);

//Grabamos el evento final
cudaEventRecord(stop,0);
cudaEventSynchronize(stop);

//Obtenemos el tiempo de ejecucion
cudaEventElapsedTime(&timeTemp, start, stop);
timeAvg = timeAvg + timeTemp;
}


//Imprimimos la media de tiempo de ejecucion
printf("Tiempo con streams: %f\n", timeAvg/100.0);


cudaStreamDestroy(s_1);
cudaStreamDestroy(s_2);

cudaFree(gpu_a_1);
cudaFree(gpu_b_1);
cudaFree(gpu_res_1);
cudaFree(gpu_a_2);
cudaFree(gpu_b_2);
cudaFree(gpu_res_2);




free(vect_a);
free(vect_b);
free(vect_res);

return 0;
}
