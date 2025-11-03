#include <stdio.h>
#include <stdint.h>
#include "./gifenc.h"
#include <cstdint>



typedef uint8_t sm;

#define frames 1000

//Definiremos las normas en memoria constante
__constant__ sm liveArray[9] = {0,0,1,1,0,0,0,0,0}; //1 si la celula vive, 0 si muere
__constant__ sm breedArray[9] = {0,0,1,1,0,0,0,0,0}; //1 si la celula nace, 0 si no

__device__ int conwaySum(int *sum, int x, int y, int mode, cudaTextureObject_t tex){
int rows = blockDim.y*gridDim.y;
int cols = blockDim.x*gridDim.x;

//Calculamos segun el lugar donde se encuentra la célula
switch(mode){
case 0:
	*sum =  
	tex2D<sm>(tex, (x - 1), 		y) +
	tex2D<sm>(tex, (x + 1) % cols , 	y) +
	tex2D<sm>(tex, x, 			(y - 1)) +
	tex2D<sm>(tex, (x - 1), 		(y - 1)) +
	tex2D<sm>(tex, (x + 1) % cols , 	(y - 1)) +
	tex2D<sm>(tex, x,			(y + 1) % rows) +
	tex2D<sm>(tex, (x - 1),			(y + 1) % rows) +
	tex2D<sm>(tex, (x + 1) % cols , 	(y + 1) % rows);
break;

case 1:
	*sum =  
	tex2D<sm>(tex, cols - 1,		y) +
	tex2D<sm>(tex, (x + 1) % cols , 	y) +
	tex2D<sm>(tex, x, 			(y - 1)) +
	tex2D<sm>(tex, cols - 1, 		(y - 1)) +
	tex2D<sm>(tex, (x + 1) % cols , 	(y - 1)) +
	tex2D<sm>(tex, x,			(y + 1) % rows) +
	tex2D<sm>(tex, cols - 1,		(y + 1) % rows) +
	tex2D<sm>(tex, (x + 1) % cols , 	(y + 1) % rows);
break;

case 2:
	*sum =  
	tex2D<sm>(tex, (x - 1),			y) +
	tex2D<sm>(tex, (x + 1) % cols , 	y) +
	tex2D<sm>(tex, x, 			rows - 1) +
	tex2D<sm>(tex, (x - 1),			rows - 1) +
	tex2D<sm>(tex, (x + 1) % cols , 	rows - 1) +
	tex2D<sm>(tex, x,			(y + 1) % rows) +
	tex2D<sm>(tex, (x - 1),			(y + 1) % rows) +
	tex2D<sm>(tex, (x + 1) % cols , 	(y + 1) % rows);
break;

case 3:
	*sum =  
	tex2D<sm>(tex, cols - 1,		y) +
	tex2D<sm>(tex, (x + 1) % cols , 	y) +
	tex2D<sm>(tex, x, 			rows - 1) +
	tex2D<sm>(tex, cols - 1, 		rows - 1) +
	tex2D<sm>(tex, (x + 1) % cols , 	rows - 1) +
	tex2D<sm>(tex, x,			(y + 1) % rows) +
	tex2D<sm>(tex, cols - 1,		(y + 1) % rows) +
	tex2D<sm>(tex, (x + 1) % cols , 	(y + 1) % rows);
break;

}


}


__global__ void conway(sm *output, cudaTextureObject_t tex){
int sum;
int x = threadIdx.x+blockDim.x*blockIdx.x;
int y = threadIdx.y+blockDim.y*blockIdx.y;
int edge = 0;

//En caso de estar en estos bordes no basta con la división para conseguir todos los valores, asi que los tratamos por separado
if(x == 0) edge = edge + 1;
if(y == 0) edge = edge + 2;

//Llamamos a la funcion suma para ver el numero de celulas vivas a su alrededor
conwaySum(&sum, x, y, edge, tex);

//Decidimos el estado de la celula segun las normas que hemos indicado
if (tex2D<sm>(tex, x, y) == 0 && breedArray[sum] == 1){
output[x + y * blockDim.x * gridDim.x] = 1;
}

else if (tex2D<sm>(tex, x, y) == 1 && liveArray[sum] == 0){  
output[x + y * blockDim.x * gridDim.x] = 0;
}
}






int main(void){
//Decidimos las dimensiones del tablero
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, 0);
cudaTextureObject_t tex;
const int num_rows = 32;
const int num_cols = prop.texturePitchAlignment*1;
const int ts = num_cols*num_rows;
const int ds = ts*sizeof(sm);

sm *dataIn;
dataIn =(sm*) malloc(ts);

//Inicializamos un tablero vacio y un tablero de output
for (int i = 0; i < ts; i++) dataIn[i] = 0; 
sm* dataDev;
cudaMalloc((void**)&dataDev, ds);

sm* dataOut;
cudaMalloc((void**)&dataOut, ds);

//Declaramos las celulas iniciales
dataIn[num_cols*16+14] = 1;
dataIn[num_cols*16+15] = 1;
dataIn[num_cols*16+16] = 1;
dataIn[num_cols*15+16] = 1;
dataIn[15+num_cols*14] = 1;

//Creamos la textura
struct cudaResourceDesc resDesc;
memset(&resDesc, 0, sizeof(resDesc));
resDesc.resType = cudaResourceTypePitch2D;
resDesc.res.pitch2D.devPtr = dataDev;
resDesc.res.pitch2D.width = num_cols;
resDesc.res.pitch2D.height = num_rows;
resDesc.res.pitch2D.desc = cudaCreateChannelDesc<sm>();
resDesc.res.pitch2D.pitchInBytes = num_cols*sizeof(sm);
struct cudaTextureDesc texDesc;
memset(&texDesc, 0, sizeof(texDesc));
cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL);

//Pasamos los datos iniciales a la textura y al output
cudaMemcpy(dataDev, dataIn, ds, cudaMemcpyHostToDevice);
cudaMemcpy(dataOut, dataIn, ds, cudaMemcpyHostToDevice);



//Declaramos el tamaño de nuestros bloques y el total
dim3 threadSize(32,32);
dim3 blockSize(num_cols/threadSize.x,num_rows/threadSize.y);



int w = num_cols, h = num_rows;

//Inicializamos el gif con el paquete gifenc
ge_GIF *gif;
    
uint8_t palette[] = {
    0x00, 0x00, 0x00, // 0 -> black
    0xFF, 0xFF, 0xFF, // 1 -> white
    };
    
gif = ge_new_gif(
        "conway_23_23.gif",  /* file name */
        w, h,           /* canvas size */
        palette,
        1,              /* palette depth == log2(# of colors) */
        -1,             /* no transparency */
        0               /* infinite loop */
	);
//Creamos el frame inicial y comenzamos a iterar
gif->frame =  dataIn;
ge_add_frame(gif, 5);
for (int i = 0; i < frames; i++) {
	printf("frame = %d\n",i);
	
   	//Procesamos el siguiente frame
    	conway<<<blockSize, threadSize>>>(dataOut, tex);
    	cudaDeviceSynchronize();
    	
    	//Damos los datos del frame nuevo a la textura
    	cudaMemcpy(dataDev, dataOut, ds, cudaMemcpyDeviceToDevice);
    	
	//Lo extraemos y añadimos al gif
	cudaMemcpy(gif->frame, dataDev, ds, cudaMemcpyDeviceToHost); 		
    	cudaDeviceSynchronize();
    	ge_add_frame(gif, 5);
    	}
//Cerramos el gif
ge_close_gif(gif);


cudaFree(dataDev);
cudaFree(dataOut);
cudaFree(liveArray);
cudaFree(breedArray);
free(dataIn);

return 0;
}
