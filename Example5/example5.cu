#include <stdio.h>

#define  N  1000

struct Matrix{
int fil;
int col;
double* data;
};


__global__ void sqr_matrix_mult(double *matrix_a, double *matrix_b, double *matrix_result, int size){

//Calculamos nuestro indice segun el bloque en el que estamos y nuestra
//posicion dentro de este
int i = threadIdx.x+blockDim.x + blockIdx.x;
int j = threadIdx.y+blockDim.y + blockIdx.y;

//Comprobamos que seguimos dentro de los limites de la matriz
if (i < size && j < size){
	int index = i*size+j;
	matrix_result[index] = 0.0;

	for (int iter = 0; iter < size; iter++ ) {

		matrix_result[index] = matrix_result[index]
		+ matrix_a[size*i+iter] * matrix_b[size*iter+j];
		}	
	}
}


int main(void){

Matrix A, B, C;
double *gpu_A, *gpu_B, *gpu_C;

dim3 threadSize(64,64);
dim3 blockSize(N/threadSize.x + 1,N/threadSize.y + 1);

A.fil = N; A.col = N;
A.data =(double*) malloc(sizeof(double)*N*N);

B.fil = N; B.col = N;
B.data = (double*) malloc(sizeof(double)*N*N);

C.fil = N; C.col = N;
C.data = (double*) malloc(sizeof(double)*N*N);

//Reservamos memoria en la GPU
cudaMalloc((void**) &gpu_A, N*N*sizeof(double));
cudaMalloc((void**) &gpu_B, N*N*sizeof(double));
cudaMalloc((void**) &gpu_C, N*N*sizeof(double));

//Rellenamos las matrices A y B
for (int i=0; i<N; i++){
	for (int j=0; j<N; j++){
		A.data[i*N+j]=i+j;
		B.data[i*N+j]=i-j;
}
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
for(int i=0; i<1000; i++){

printf("iter: %d\n", i);

//Grabamos el evento inicial
cudaEventRecord(start,0);

//Enviamos nuestras matrices a la GPU
cudaMemcpy(gpu_A, A.data, N*N*sizeof(double), cudaMemcpyHostToDevice);
cudaMemcpy(gpu_B, B.data, N*N*sizeof(double), cudaMemcpyHostToDevice);

//Lanzamos nuestra función
sqr_matrix_mult<<<blockSize,threadSize>>>(gpu_A,gpu_B,gpu_C,N);

//Extraemos nuestro resultado de la GPU
cudaMemcpy(C.data, gpu_C, N*N*sizeof(double), cudaMemcpyDeviceToHost);

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
printf("Tiempo: %f\n", timeAvg/1000.0);

//Liberamos la memoria
cudaFree(gpu_A);
cudaFree(gpu_B);
cudaFree(gpu_C);
free(A.data);
free(B.data);
free(C.data);
return 0;
}
