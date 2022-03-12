#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <cuda_runtime.h>
#include "common.h"

#define NB_STREAMS 8

cudaStream_t streams[NB_STREAMS];

void allocate_device_MPI_process(int rank) {
    int nbGPU, deviceUsed;

    cudaGetDeviceCount(&nbGPU);
    cudaCheckErrors(cudaSetDevice(rank % nbGPU));
    cudaCheckErrors(cudaGetDevice(&deviceUsed));

    printf("MPI process %d uses device %d\n", rank, deviceUsed);
}

void createCudaStreams() {
    for(int i = 0; i < NB_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }
}

void destroyCudaStreams() {
    for(int i = 0; i < NB_STREAMS; i++) {
        cudaStreamDestroy(streams[i]);
    }
}

bool verify_threshold(pixel* source, pixel* dest, int j, int k, int width, int threshold) {
    float diff_r;
    float diff_g;
    float diff_b;

    diff_r = (dest[CONV(j, k, width)].r - source[CONV(j, k, width)].r);
    diff_g = (dest[CONV(j, k, width)].g - source[CONV(j, k, width)].g);
    diff_b = (dest[CONV(j, k, width)].b - source[CONV(j, k, width)].b);

    if (diff_r > threshold || -diff_r > threshold
        ||
        diff_g > threshold || -diff_g > threshold
        ||
        diff_b > threshold || -diff_b > threshold
            ) {
        return false;
    }
    return true;
}

__global__ void bool_and_multi_block(const bool* in, int array_size, bool* out) {
    int thIdx = threadIdx.x;
    int gthIdx = thIdx + blockIdx.x * blockDim.x;
    const int gridSize = gridDim.x * blockDim.x;

    bool result = true;
    for (int i = gthIdx; i < array_size; i += gridSize) {
        result = result && in[i];
    }

    __shared__ bool shared_arr[blockDim.x];
    shared_arr[thIdx] = result;
    __syncthreads();

    for (int size = blockDim.x / 2; size > 0; size /= 2) {
        if (thIdx < size) {
            shared_arr[thIdx] = shared_arr[thIdx] && shared_arr[thIdx + size];
        }
        __syncthreads();
    }

    if (thIdx == 0) {
        out[blockIdx.x] = shared_arr[0];
    }
}

__global__ void apply_blur_filter_loop_srcToDest(pixel* source, pixel* dest, int width, int threshold,
                                                 int minCol, int maxCol, int minRow, int maxRow, bool* end) {
    int j, k;

    j = blockIdx.x * blockDim.x + threadIdx.x;
    k = blockIdx.y * blockDim.y + threadIdx.y;

    if(minRow <= j && j < maxRow) {
        if(minCol <= k && k < maxRow) {
            dest[CONV(j, k, width)].r = source[CONV(j, k, width)].r;
            dest[CONV(j, k, width)].g = source[CONV(j, k, width)].g;
            dest[CONV(j, k, width)].b = source[CONV(j, k, width)].b;

            end[CONV(j, k, width)] = verify_threshold(source, dest, j, k, width, threshold);
        }
    }
}

__global__ void apply_blur_filter_loop_medium(pixel* source, pixel* dest, int width, int size, int threshold,
                                              int minCol, int maxCol, int minRow, int maxRow, bool* end) {
    int j, k;

    j = blockIdx.x * blockDim.x + threadIdx.x;
    k = blockIdx.y * blockDim.y + threadIdx.y;

    if(minRow <= j && j < maxRow) {
        if(minCol <= k && k < maxRow) {
            int stencil_j, stencil_k;
            int t_r = 0;
            int t_g = 0;
            int t_b = 0;

            for (stencil_j = -size; stencil_j <= size; stencil_j++) {
                for (stencil_k = -size; stencil_k <= size; stencil_k++) {
                    t_r += source[CONV(j + stencil_j, k + stencil_k, width)].r;
                    t_g += source[CONV(j + stencil_j, k + stencil_k, width)].g;
                    t_b += source[CONV(j + stencil_j, k + stencil_k, width)].b;
                }
            }

            dest[CONV(j, k, width)].r = t_r / ((2 * size + 1) * (2 * size + 1));
            dest[CONV(j, k, width)].g = t_g / ((2 * size + 1) * (2 * size + 1));
            dest[CONV(j, k, width)].b = t_b / ((2 * size + 1) * (2 * size + 1));


            end[CONV(j, k, width)] = verify_threshold(source, dest, j, k, width, threshold);
        }
    }
}

__global__ void apply_blur_filter_loop_final(pixel* source, pixel* dest, int threshold, int width,
                                             int minCol, int maxCol, int minRow, int maxRow, char* end) {
    int j, k;

    j = blockIdx.x * blockDim.x + threadIdx.x;
    k = blockIdx.y * blockDim.y + threadIdx.y;

    if(minRow <= j && j < maxRow) {
        if (minCol <= k && k < maxCol) {
            float diff_r;
            float diff_g;
            float diff_b;

            diff_r = (dest[CONV(j, k, width)].r - source[CONV(j, k, width)].r);
            diff_g = (dest[CONV(j, k, width)].g - source[CONV(j, k, width)].g);
            diff_b = (dest[CONV(j, k, width)].b - source[CONV(j, k, width)].b);

            if (diff_r > threshold || -diff_r > threshold
                ||
                diff_g > threshold || -diff_g > threshold
                ||
                diff_b > threshold || -diff_b > threshold
                    ) {
                end[CONV(j, k, width)] = 0;
            }

            source[CONV(j, k, width)].r = dest[CONV(j, k, width)].r;
            source[CONV(j, k, width)].g = dest[CONV(j, k, width)].g;
            source[CONV(j, k, width)].b = dest[CONV(j, k, width)].b;
        }
    }
}

void swap(pixel** a, pixel** b) {
    pixel* t = *a;
    *a = *b;
    *b = t;
}

void apply_blur_filter_cuda(animated_gif *image, int size, int threshold, int image_index, striping_info* s_info) {
    int j, k;
    int width, height;
    int end = 0;
    int n_iter = 0;

    pixel *p;

    pixel *p_device;
    pixel *new_device;
    bool* end_device;
    bool* end_reduced_device;

    /* Get the pixels of all images */
    p = (image->p)[image_index];


    /* Process all images */
    n_iter = 0;
    width = image->width[image_index];
    height = image->height[image_index];

    dim3 blockSize(16, 16);
    int gridSize = (width * height) / 256 + 1;
    const int reducedEndSize = 64;

    checkCudaErrors(cudaMalloc((void**) &p_device, width * height * sizeof(pixel), streams[0]));
    checkCudaErrors(cudaMalloc((void**) &new_device, width * height * sizeof(pixel), streams[1]));
    checkCudaErrors(cudaMalloc((void**) &end_device, width * height * sizeof(bool), streams[2]));
    checkCudaErrors(cudaMalloc((void**) &end_reduced_device, reducedEndSize * sizeof(bool), streams[3]));

    checkCudaErrors(cudaMemcpy(p_device, p, width * height * sizeof(pixel), cudaMemcpyHostToDevice, streams[0]));

    for (int i = 0; i < 4; ++i) {
        cudaCheckErrors(cudaStreamSynchronize(streams[i]));
    }

    do {
        end = 1;
        n_iter++;

        const int begin1    = max(0, s_info->min_row);
        const int end1      = min(size, s_info->max_row);
        const int begin2    = max(size, s_info->min_row);
        const int end2      = min(height - size, s_info->max_row);
        const int begin3    = max(size, s_info->min_row);
        const int end3      = min(height - size, s_info->max_row);
        const int begin4    = max(height - size, s_info->min_row);
        const int end4      = min(height, s_info->max_row);
        const int begin5    = max(size, s_info->min_row);
        const int end5      = min(height / 10 - size, s_info->max_row);
        const int begin6    = max(height / 10 - size, s_info->min_row);
        const int end6      = min((int)(height * 0.9 + size), s_info->max_row);
        const int begin7    = max((int)(height * 0.9 + size), s_info->min_row);
        const int end7      = min(height - size, s_info->max_row);
        const int begin8    = max(1, s_info->min_row);
        const int end8      = min(height, s_info->max_row);

        apply_blur_filter_loop_srcToDest<<<gridSize, blockSize, stream[0]>>>(
                p_device, new_device, width, threshold,
                begin1, end1, 0, width, end_device
                );

        apply_blur_filter_loop_srctodest<<<gridsize, blocksize, stream[1]>>>(
                p_device, new_device, width, threshold,
                begin2, end2, 0, size, end_device
        );

        apply_blur_filter_loop_srcToDest<<<gridSize, blockSize, stream[2]>>>(
                p_device, new_device, width, threshold,
                begin3, end3, width - size, width, end_device
        );

        apply_blur_filter_loop_srcToDest<<<gridSize, blockSize, stream[3]>>>(
                p_device, new_device, width, threshold,
                begin4, end4, 0, width, end_device
        );

        apply_blur_filter_loop_medium<<<gridSize, blockSize, stream[4]>>>(
               p_device, new_device, width, size, threshold,
               begin5, end5, size, width - size, end_device
        );

        apply_blur_filter_loop_srcToDest<<<gridSize, blockSize, stream[5]>>>(
                p_device, new_device, width, threshold,
                begin6, end6, size, width - size, end_device
        );

        apply_blur_filter_loop_medium<<<gridSize, blockSize, stream[6]>>>(
                p_device, new_device, width, size, threshold,
                begin7, end7, size, width - size, end_device
        );

        for (int i = 0; i < 7; ++i) {
            cudaCheckErrors(cudaStreamSynchronize(streams[i]));
        }

        bool_and_multi_block<<<reducedEndSize, gridSize>>>(end_device, width * height, end_reduced_device);
        bool_and_multi_block<<<1, gridSize>>>(end_reduced_device, reducedEndSize, end_reduced_device);

        checkCudaErrors(cudaMemcpy(&end, end_reduced_device, sizeof(bool), cudaMemcpyDeviceToHost));

        swap(&p_device, &new_device);
    } while(threshold > 0 && !end);

    cudaCheckErrors(cudaMemcpy(p, p_device, width * height * sizeof(pixel), cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaFree(p_device));
    checkCudaErrors(cudaFree(new_device));
    checkCudaErrors(cudaFree(end_device));
    checkCudaErrors(cudaFree(end_reduced_device));
}
