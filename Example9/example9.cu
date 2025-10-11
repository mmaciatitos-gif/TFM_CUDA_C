#include <stdio.h>

#define  N  1000000
#define threadSize 64

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
res[0]= res[0]+cache[0];
}

__global__ void productoEscalarAtomicAll(float *vector_a, float *vector_b, float *res){
const int index = threadIdx.x + blockIdx.x*blockDim.x;
atomicAdd(&res[0], vector_a[index]*vector_b[index]);
}



int main(void){

//Definimos las variables principales
float *vector_a, *vector_b;
float *gpu_vector_a, *gpu_vector_b;
float *gpu_result;
float result;

//Reservamos espacio para los vectores 
vector_a = (float*) malloc(sizeof(float)*N);
vector_b = (float*) malloc(sizeof(float)*N);

//Para poder dividir el trabajo en bloques del mismo tamaño ajustamos las dimensiones del vector sobre la GPU para ser un multiplo de su tamaño
//dado que nuestros bloques tendran un tamaño multiplo de 2 facilitamos tambien asi el codigo del kernel
int padded_N;

if(N%threadSize==0)
padded_N = N;
else
padded_N = (N/threadSize+1)*threadSize;

//Reservamos espacio en la GPU para los vectores y el resultado
cudaMalloc((void**) &gpu_vector_a, padded_N*sizeof(float));
cudaMalloc((void**) &gpu_vector_b, padded_N*sizeof(float));
cudaMalloc((void**) &gpu_result, sizeof(float));

//Rellenamos las variables en la GPU con 0s para evitar posibles problemas
cudaMemset(gpu_vector_a,0,padded_N*sizeof(float));
cudaMemset(gpu_vector_b,0,padded_N*sizeof(float));
cudaMemset(gpu_result,0,sizeof(float));

//Rellenamos los vectores sobre la CPU
for(int i = 0; i<N; i++){
vector_a[i] = 1.0;
vector_b[i] = 1.0;
}

//Enviamos los vectores a la GPU
cudaMemcpy(gpu_vector_a, vector_a, N*sizeof(float),cudaMemcpyHostToDevice);
cudaMemcpy(gpu_vector_b, vector_b, N*sizeof(float),cudaMemcpyHostToDevice);

//Calculamos primero el valor real con el kernel atomico
productoEscalarAtomic<<<padded_N/threadSize,threadSize>>>(gpu_vector_a,gpu_vector_b,gpu_result);

//Extraemos el resultado
cudaMemcpy(&result, &gpu_result[0], sizeof(float),cudaMemcpyDeviceToHost);

//Imprimimos el resultado
printf("Resultado atómico = %f\n", result);

//Reseteamos la variable resultado
cudaMemset(gpu_result,0,sizeof(float));

//Calculamos ahora con el kernel sin operación atómica
productoEscalarNoAtomic<<<padded_N/threadSize,threadSize>>>(gpu_vector_a,gpu_vector_b,gpu_result);

//Extraemos el resultado
cudaMemcpy(&result, &gpu_result[0], sizeof(float),cudaMemcpyDeviceToHost);

//Imprimimos el resultado
printf("Resultado no atómico = %f\n", result);

//Reseteamos la variable resultado
cudaMemset(gpu_result,0,sizeof(float));


//Creamos las variables evento
cudaEvent_t  start, stop;

//Creamos los eventos
cudaEventCreate (&start);
cudaEventCreate (&stop);

//Creamos una variable para almacenar los tiempos y otra para hacer media:
float timeTemp = 0;
float timeAvg;

//########################################################################################################################

//Kernel original

//Reseteamos el tiempo medio
timeAvg = 0;

//Creamos un bucle
for(int i=0; i<100; i++){

//Grabamos el evento inicial
cudaEventRecord(start,0);

//Calculamos con el kernel original
productoEscalarAtomic<<<padded_N/threadSize,threadSize>>>(gpu_vector_a,gpu_vector_b,gpu_result);

//Grabamos el evento final
cudaEventRecord(stop,0);
cudaEventSynchronize(stop);

//Obtenemos el tiempo de ejecucion
cudaEventElapsedTime(&timeTemp, start, stop);
timeAvg = timeAvg + timeTemp;
}


//Imprimimos la media de tiempo de ejecucion
printf("Tiempo original: %f\n", timeAvg/100.0);



//########################################################################################################################

//Kernel modificado

//Reseteamos el tiempo medio
timeAvg = 0;

//Creamos un bucle
for(int i=0; i<100; i++){

//Grabamos el evento inicial
cudaEventRecord(start,0);

//Calculamos con el kernel modificado
productoEscalarAtomicAll<<<padded_N/threadSize,threadSize>>>(gpu_vector_a,gpu_vector_b,gpu_result);

//Grabamos el evento final
cudaEventRecord(stop,0);
cudaEventSynchronize(stop);

//Obtenemos el tiempo de ejecucion
cudaEventElapsedTime(&timeTemp, start, stop);
timeAvg = timeAvg + timeTemp;
}


//Imprimimos la media de tiempo de ejecucion
printf("Tiempo modificado: %f\n", timeAvg/100.0);











free(vector_a);
free(vector_b);
cudaFree(gpu_vector_a);
cudaFree(gpu_vector_b);
cudaFree(gpu_result);
}
