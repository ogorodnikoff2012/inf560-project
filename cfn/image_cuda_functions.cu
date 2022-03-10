#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <cuda_runtime.h>

#define NB_STREAMS 16

#define CONV(l, c, nb_c) \
    ((l)*(nb_c)+(c))

cudaStream_t streams[NB_STREAMS];

device volatile int t_r = 0;
device volatile int t_g = 0;
device volatile int t_b = 0;

void allocate_device_MPI_process(int rank) {
    int nbGPU, deviceUsed;

    cudaGetDeviceCount(&nbGPU);
    cudaCheckErrors(cudaSetDevice(rank % nbGPU));
    cudaCheckErrors(cudaGetDevice(&deviceUsed));

    for(int i = 0; i < NB_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }

    printf("MPI process %d uses device %d\n", rank, deviceUsed);
}

void destroyCudaStreams() {
    for(int i = 0; i < NB_STREAMS; i++) {
        cudaStreamDestroy(streams[i]);
    }
}

__global__ void blur_filter_cuda_kernel(int size, pixel* p, int j, int k, int width) {
    int stencil_j, stencil_k;

    stencil_j = blockIdx.x * blockDim.x + threadIdx.x;
    stencil_k = blockIdx.y * blockDim.y + threadIdx.y;

    if(i >= -size && i <= size) {
        if(j >= -size && j <= size) {
            t_r += p[CONV(j + stencil_k, k + stencil_k, width)].r;
            t_g += p[CONV(j + stencil_k, k + stencil_k, width)].g;
            t_b += p[CONV(j + stencil_k, k + stencil_k, width)].b;
        }
    }

    __syncthreads();
}

int* blur_filter_cuda(int omprank, int size, pixel* p, int j, int k, int width) {
    pixel* d_p;
    int result[3];
    checkCudaErrors(cudaMalloc((void**) &d_p, p, size * sizeof(pixel), cudaMemcpyHostToDevice, streams[omprank % NB_STREAMS]));
    checkCudaErrors(cudaMemcpyFromSymbol(&result[0], t_r, sizeof(int)));
    checkCudaErrors(cudaMemcpyFromSymbol(&result[1], t_b, sizeof(int)));
    checkCudaErrors(cudaMemcpyFromSymbol(&result[2], t_b, sizeof(int)));
    cudaFree(d_p);
    return result;
}