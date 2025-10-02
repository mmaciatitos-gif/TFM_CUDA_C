#include <stdio.h>
#include <omp.h>
#include <stdlib.h>

#define  N  1000

struct Matrix{
int fil;
int col;
double* data;
};


void sqr_matrix_mult(struct Matrix matrix_A, struct Matrix matrix_B, struct Matrix matrix_res);


int main(void){

struct Matrix A, B, C;

A.fil = N; A.col = N;
A.data =(double*) malloc(sizeof(double)*N*N);

B.fil = N; B.col = N;
B.data = (double*) malloc(sizeof(double)*N*N);

C.fil = N; C.col = N;
C.data = (double*) malloc(sizeof(double)*N*N);

//Rellenamos las matrices A y B
for (int i=0; i<N; i++){
	for (int j=0; j<N; j++){
		A.data[i*N+j]=i+j;
		B.data[i*N+j]=i-j;
}
}

 double Tinicial, Tfinal, Ttotal;

Ttotal = 0.0;
for(int i = 0; i<100; i++){

printf("iter=%d\n", i);

Tinicial = omp_get_wtime();


//Lanzamos nuestra función
sqr_matrix_mult(A,B,C);
Tfinal = omp_get_wtime();

Ttotal = Ttotal + Tfinal - Tinicial;
}

printf("Tiempo: %f\n", Ttotal/100);


//Liberamos la memoria
free(A.data);
free(B.data);
free(C.data);
return 0;
}

void sqr_matrix_mult(struct Matrix matrix_A, struct Matrix matrix_B, struct Matrix matrix_res){
int i, j, iter, iter2;
int size = matrix_A.fil;

#pragma omp parallel for
for(iter = 0; iter < size*size; iter++) {
    matrix_res.data[iter]=0.0;
    i = iter/size;
    j = iter%size;
    for(iter2=0; iter2 < size; iter2++){
      matrix_res.data[iter]= matrix_res.data[iter]+
      + matrix_A.data[size*i+iter2] * matrix_B.data[size*iter2+j];
    }
}

}

