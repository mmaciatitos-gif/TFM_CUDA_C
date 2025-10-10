#include <stdio.h>

#define  N  10000
#define  threadSize 32

__global__ void productoEscalar(float *vector_a, float *vector_b, float *res){

//Creamos la variable compartida, donde almacenamos la seccion de producto escalar asignada al bloque
__shared__ float cache[threadSize];

//Definimos un iterador y una variable que indica que elemento del vector corresponde al hilo
int iter = blockDim.x;
int index = threadIdx.x + blockIdx.x*blockDim.x;

//Multiplicamos los elementos de los vectores en el indice asociado al hilo y lo guardamos en la variable compartida
cache[threadIdx.x] = vector_a[index]*vector_b[index];

//Nos aseguramos de que todos los productos se han realizado antes de pasar a la suma
__syncthreads();

while(iter > 1){

//Sumamos la primera mitad del vector con la segunda y la almacenamos en la primera
//hecho esto, tratamos la primera mitad como el vector completo y repetimos hasta quedar un unico elemento
if(threadIdx.x < iter/2){
cache[threadIdx.x] = cache[threadIdx.x] + cache[threadIdx.x + iter/2];
}
iter = iter/2;

//En cada paso aseguramos que hemos realizado todas las sumas antes de pasar a la siguiente iteracion
__syncthreads();
}

//Utilizamos una operacion atomica para sumar los resultados de cada bloque en una unica variable
if(threadIdx.x == 0)
atomicAdd(&res[0], cache[0]);
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

printf("N/threadsize * threadsize = %d\n", padded_N/64);

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
vector_a[i] = 1.0/(i+1.0);
vector_b[i] = 1.0+i;
}

//Enviamos los vectores a la GPU
cudaMemcpy(gpu_vector_a, vector_a, N*sizeof(float),cudaMemcpyHostToDevice);
cudaMemcpy(gpu_vector_b, vector_b, N*sizeof(float),cudaMemcpyHostToDevice);

//Lanzamos nuestro kernel
productoEscalar<<<padded_N/threadSize,threadSize>>>(gpu_vector_a,gpu_vector_b,gpu_result);

//Extraemos el resultado
cudaMemcpy(&result, &gpu_result[0], sizeof(float),cudaMemcpyDeviceToHost);

//Imprimimos el resultado
printf("result=%f\n", result);

//Liberamos las variables
free(vector_a);
free(vector_b);
cudaFree(gpu_vector_a);
cudaFree(gpu_vector_b);
cudaFree(gpu_result);

return 0;
}
