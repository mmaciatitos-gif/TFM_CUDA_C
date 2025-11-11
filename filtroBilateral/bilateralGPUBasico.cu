// Compilar: gcc  -o bilateral bilateral.c -lm
// Ejecutar: ./bilateral lena_ruido005.jpg output.png

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

__constant__ int widthGPU, heightGPU, channelsGPU;


__device__ float gaussian(float x, float sigma) {
    return expf(-(x * x) / (2.0f * sigma * sigma));
}

__global__ void normalize(unsigned char *input, float *output){
	int ind = threadIdx.x +blockIdx.x*blockDim.x;
	int len = widthGPU*heightGPU*channelsGPU;
	if(ind < len){
	output[ind] = input[ind]/ 255.0f;
	}
}

__global__ void bilateral_adaptive(float *input, unsigned char *output) {
    int x = threadIdx.x +blockIdx.x*blockDim.x;
    int y = threadIdx.y +blockIdx.y*blockDim.y;
    int c = blockIdx.z;
    int i, j;
    if (y < heightGPU) {
        if (x < widthGPU) {                 
                float sum = 0.0f;
                float wsum = 0.0f;
                float center_val = input[(y * widthGPU + x) * channelsGPU + c];

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
                        float val = input[(yy * widthGPU + xx) * channelsGPU + c];
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
                        float val = input[(yy * widthGPU + xx) * channelsGPU + c];
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

                        float val = input[(yy * widthGPU + xx) * channelsGPU + c];

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
    
    unsigned char *outputGPU;
    cudaMalloc((void**)&outputGPU, width * height * channels);
    
    unsigned char *inputGPU;
    cudaMalloc((void**)&inputGPU, width * height * channels);	

    float *normGPU;
    cudaMalloc((void**)&normGPU, width * height * channels*sizeof(float));
    
    cudaMemcpyToSymbol(widthGPU, &width, sizeof(int));
    cudaMemcpyToSymbol(heightGPU, &height, sizeof(int));
    cudaMemcpyToSymbol(channelsGPU, &channels, sizeof(int));

	cudaMemcpy(inputGPU, img, width * height * channels, cudaMemcpyHostToDevice);
	normalize<<<(width * height * channels)/64+1, 64>>>(inputGPU, normGPU);
	
	dim3 threadSize(16,16);
	dim3 blockSize(width/threadSize.x + 1,height/threadSize.y + 1, channels);
	bilateral_adaptive<<<blockSize, threadSize>>>(normGPU, outputGPU);
	
	cudaMemcpy(output, outputGPU, width * height * channels, cudaMemcpyDeviceToHost);



    if (!stbi_write_png(argv[2], width, height, channels, output, width * channels)) {
        printf("Error escribiendo la imagen\n");
        free(img);
        free(output);
        return 1;
    }

    printf("Filtro bilateral adaptativo aplicado con exito.\n");

    free(img);
    free(output);
    return 0;
}

