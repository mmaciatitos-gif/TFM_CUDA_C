#include <stdio.h>

#define  N  10000

__constant__ int add_list_ct[N];

__global__ void kernel_gl_mem(int *list, int *add_list){
const int index = blockIdx.x + threadIdx.x*blockDim.x;
if(index<N){
	for(int i = 0; i<N; i++){
		list[index] = list[index] + add_list[i];
}}}

__global__ void kernel_reg_mem(int *list, int *add_list){
const int index = blockIdx.x + threadIdx.x*blockDim.x;
if(index<N){
	int value = list[index];
	for(int i = 0; i<N; i++){
		value = value + add_list[i];
}
list[index] = value;
}}

int main(void){

int *list, *add_list, *list_res;
int *gpu_list, *gpu_add_list;

list = (int*) malloc(sizeof(int)*N);
add_list = (int*) malloc(sizeof(int)*N);
list_res = (int*) malloc(sizeof(int)*N);

//Reservamos memoria en la GPU
cudaMalloc((void**) &gpu_list, N*sizeof(int));
cudaMalloc((void**) &gpu_add_list, N*sizeof(int));

for(int i = 0; i<N; i++){
list[i] = rand() % 10;
add_list[i] = rand() % 10;
}

//Creamos las variables evento
cudaEvent_t  start, stop;

//Creamos los eventos
cudaEventCreate (&start);
cudaEventCreate (&stop);

//Creamos una variable para almacenar los tiempos y otra para hacer media:
float timeTemp = 0;
float timeAvg = 0;

//Creamos un bucle
for(int i=0; i<100; i++){

//printf("iter: %d\n", i);

//Grabamos el evento inicial
cudaEventRecord(start,0);

//Enviamos nuestras matrices a la GPU
cudaMemcpy(gpu_list, list, N*sizeof(int), cudaMemcpyHostToDevice);
cudaMemcpy(gpu_add_list, add_list, N*sizeof(int), cudaMemcpyHostToDevice);

//Lanzamos nuestra función
kernel_gl_mem<<<N/128+1,128>>>(gpu_list, gpu_add_list);

//Extraemos nuestro resultado de la GPU
cudaMemcpy(list_res, gpu_list, N*sizeof(int), cudaMemcpyDeviceToHost);

//Grabamos el evento final
cudaEventRecord(stop,0);
cudaEventSynchronize(stop);

//Obtenemos el tiempo de ejecucion
cudaEventElapsedTime(&timeTemp, start, stop);
timeAvg = timeAvg + timeTemp;
}


//Imprimimos la media de tiempo de ejecucion
printf("Tiempo global: %f\n", timeAvg/100.0);






//Creamos una variable para almacenar los tiempos y otra para hacer media:
timeAvg = 0;

//Creamos un bucle
for(int i=0; i<100; i++){

//printf("iter: %d\n", i);

//Grabamos el evento inicial
cudaEventRecord(start,0);

//Enviamos nuestras matrices a la GPU
cudaMemcpy(gpu_list, list, N*sizeof(int), cudaMemcpyHostToDevice);
cudaMemcpy(gpu_add_list, add_list, N*sizeof(int), cudaMemcpyHostToDevice);

//Lanzamos nuestra función
kernel_reg_mem<<<N/128+1,128>>>(gpu_list, gpu_add_list);

//Extraemos nuestro resultado de la GPU
cudaMemcpy(list_res, gpu_list, N*sizeof(int), cudaMemcpyDeviceToHost);

//Grabamos el evento final
cudaEventRecord(stop,0);
cudaEventSynchronize(stop);

//Obtenemos el tiempo de ejecucion
cudaEventElapsedTime(&timeTemp, start, stop);
timeAvg = timeAvg + timeTemp;
}

//Destruimos los eventos
cudaEventDestroy(start);
cudaEventDestroy(stop);

//Imprimimos la media de tiempo de ejecucion
printf("Tiempo registros: %f\n", timeAvg/100.0);


//Liberamos la memoria
cudaFree(gpu_list);
cudaFree(gpu_add_list);
free(list);
free(add_list);
return 0;
}
