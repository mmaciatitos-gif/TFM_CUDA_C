// Compilar: nvcc bilateralGPUTexturas.cu -o bilateralGPUTexturas
// Ejecutar: ./bilateralGPUTexturas lena_ruido005.jpg output.png

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Parámetros del filtro
#define KERNEL_RADIUS 3   // radio de la ventana
#define SIGMA_S 3.0       // sigma espacial
#define SIGMA_R 0.05       // sigma rango (intensidad, 0..1)

//Definiremos las dimensiones de la imagen en memoria constante
__constant__ int widthGPU, heightGPU, channelsGPU, textWidth;

__device__ float gaussian(float x, float sigma) {
    return expf(-(x * x) / (2.0f * sigma * sigma));
}

__global__ void normalizeAndSplit(unsigned char *input, float **output){
	int x = threadIdx.x+blockDim.x*blockIdx.x;
	int y = threadIdx.y+blockDim.y*blockIdx.y;
	int c = blockIdx.z;
	
	if (y < heightGPU) {
        	if (x < widthGPU) {
			output[c][x+y*textWidth] = input[(x+y*widthGPU)*channelsGPU+c]/ 255.0f;
	}}
}

__global__ void bilateral_adaptive(cudaTextureObject_t *input, unsigned char *output) {
    	int i, j;	
	int x = threadIdx.x+blockDim.x*blockIdx.x;
	int y = threadIdx.y+blockDim.y*blockIdx.y;
	int c = blockIdx.z;
    	
    	if (y < heightGPU) {
        	if (x < widthGPU) {
    		
                float sum = 0.0f;
                float wsum = 0.0f;
		float center_val = tex2D<float>(input[c], x, y);
		
                // Calcular sigma_r adaptativo según la varianza local
                float local_mean = 0.0f;
                float local_var = 0.0f;
                int count = 0;

                for (j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++) {
                    int yy = y + j;
                    if (yy < 0 || yy >= heightGPU) continue;
                    for (i = -KERNEL_RADIUS; i <= KERNEL_RADIUS; i++) {
                        int xx = x + i;
                        if (xx < 0 || xx >= widthGPU) continue;
                        float val = tex2D<float>(input[c], xx, yy);
                        local_mean += val;
                        count++;
                    }
                }
                local_mean /= count;

                for (j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++) {
                    int yy = y + j;
                    if (yy < 0 || yy >= heightGPU) continue;
                    for (i = -KERNEL_RADIUS; i <= KERNEL_RADIUS; i++) {
                        int xx = x + i;
                        if (xx < 0 || xx >= widthGPU) continue;
                        float val = tex2D<float>(input[c], xx, yy);
                        local_var += (val - local_mean) * (val - local_mean);
                    }
                }
                local_var /= count;
//                float sigma_r_adapt = SIGMA_R * sqrtf(local_var + 1e-6f);
                float sigma_r_adapt = 2.0f * sqrtf(local_var + 1e-6f); // factor 1–5
//float                sigma_r_adapt = fminf(fmaxf(sqrtf(local_var + 1e-6f), 0.03f), 0.08f);

                // Filtro bilateral
                for (j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++) {
                    int yy = y + j;
                    if (yy < 0 || yy >= heightGPU) continue;
                    for (i = -KERNEL_RADIUS; i <= KERNEL_RADIUS; i++) {
                        int xx = x + i;
                        if (xx < 0 || xx >= widthGPU) continue;

                        float val = tex2D<float>(input[c], xx, yy);

                        float gs = gaussian(sqrtf(i*i + j*j), SIGMA_S);
                        float gr = gaussian(val - center_val, sigma_r_adapt);

                        float w = gs * gr;
                        sum += w * val;
                        wsum += w;
                    }
                }

                output[(y * widthGPU + x) * channelsGPU + c] = (unsigned char)(fminf(fmaxf(sum / wsum, 0.0f), 1.0f) * 255.0f);
              
            }
        }
    }
                
                
      


__global__ void imageWeave(float **input, unsigned char *output){
	int x = threadIdx.x +blockIdx.x*blockDim.x;
	int y =	threadIdx.y +blockIdx.y*blockDim.y;
	int z = blockIdx.z;
	
	if ((y * widthGPU + x) * channelsGPU + z < widthGPU*heightGPU*channelsGPU){
	output[(y * widthGPU + x) * channelsGPU + z] = input[z][y*widthGPU+x];
	//printf("(%d %d %d)\n%f\n", x, y, z, tex2D<float>(input[z], x, y));
	}
}






int main(int argc, char **argv) {
	if (argc < 3) {
        	printf("Uso: %s input.png output.png\n", argv[0]);
        	return 1;
    	}

    	int width, height, channels;
   	unsigned char *img = stbi_load(argv[1], &width, &height, &channels, 0);
   	if (!img) {
        	printf("Error cargando la imagen\n");
       	 	return 1;
	}

    	unsigned char *output = (unsigned char*)malloc(width * height * channels);
	
	//Creamos un input, output y enviamos las dimensiones de la imagen a la GPU
	unsigned char *outputGPU;
    	cudaMalloc((void**)&outputGPU, width * height * channels);
    
    	unsigned char *inputGPU;
    	cudaMalloc((void**)&inputGPU, width * height * channels);	
	
	
	cudaMemcpyToSymbol(widthGPU, &width, sizeof(int));
	cudaMemcpyToSymbol(heightGPU, &height, sizeof(int));
	cudaMemcpyToSymbol(channelsGPU, &channels, sizeof(int));
	
	//Creamos una textura para cada canal
	cudaTextureObject_t tex[channels];
	struct cudaResourceDesc resDesc[channels];
	float* dataDev[channels];
	struct cudaTextureDesc texDesc[channels];
	
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	const int num_rows = height;
	const int num_cols = prop.texturePitchAlignment*(width/prop.texturePitchAlignment + 1);
	const int ts = num_cols*num_rows;
	const int ds = ts*sizeof(float);
	
	
	//Creamos las texturas en el siguiente bucle
	
	for(int i = 0; i<channels; i++){
	cudaMalloc((void**)&(dataDev[i]), ds);
	//Creamos la textura

	memset(&(resDesc[i]), 0, sizeof(resDesc[i]));
	resDesc[i].resType = cudaResourceTypePitch2D;
	resDesc[i].res.pitch2D.devPtr = dataDev[i];
	resDesc[i].res.pitch2D.width = num_cols;
	resDesc[i].res.pitch2D.height = num_rows;
	resDesc[i].res.pitch2D.desc = cudaCreateChannelDesc<float>();
	resDesc[i].res.pitch2D.pitchInBytes = num_cols*sizeof(float);
	
	memset(&(texDesc[i]), 0, sizeof(texDesc[i]));
	cudaCreateTextureObject(&(tex[i]), &(resDesc[i]), &(texDesc[i]), NULL);
	}
	
	cudaMemcpyToSymbol(textWidth, &num_cols, sizeof(int));
	
	//Declaramos el tamaño de nuestros bloques y el total
	dim3 threadSize(16,16);
	dim3 blockSize(width/threadSize.x + 1,height/threadSize.y + 1, channels);
	
	//Enviamos el input y lo dividimos en texturas
	cudaMemcpy(inputGPU, img, width * height * channels, cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
	
	normalizeAndSplit<<<blockSize, threadSize>>>(inputGPU, dataDev);
	cudaDeviceSynchronize();
	
	
		
	bilateral_adaptive<<<blockSize, threadSize>>>(tex, outputGPU);	
	
	cudaMemcpy(output, outputGPU, width * height * channels, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	
	

	if (!stbi_write_png(argv[2], width, height, channels, output, width * channels)) {
        printf("Error escribiendo la imagen\n");
        	free(img);
        	free(output);
        return 1;
	}

	printf("Filtro bilateral adaptativo aplicado con exito.\n");
	
	for(int i = 0; i<channels; i++){
	cudaFree(dataDev[i]);
	}
	cudaFree(inputGPU);
	cudaFree(outputGPU);
	cudaFree(&heightGPU);
	cudaFree(&widthGPU);
	cudaFree(&channelsGPU);
	free(img);
	free(output);
	return 0;
}

