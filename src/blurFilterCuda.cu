#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <cuda_runtime.h>
#include "common.h"

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

__global__ void apply_blur_filter_loop_srcToDest(pixel* source, pixel* dest, int outer, int inter) {
    int j, k;

    j = blockIdx.x * blockDim.x + threadIdx.x;
    k = blockIdx.y * blockDim.y + threadIdx.y;

    if(j < outer)) {
        if(k < inter) {
            dest[CONV(j, k, width)].r = source[CONV(j, k, width)].r;
            dest[CONV(j, k, width)].g = source[CONV(j, k, width)].g;
            dest[CONV(j, k, width)].b = source[CONV(j, k, width)].b;
        }
    }
}

__global__ void apply_blur_filter_loop_medium(pixel* source, pixel* dest, int outer, int inter) {
    int j, k;

    j = blockIdx.x * blockDim.x + threadIdx.x;
    k = blockIdx.y * blockDim.y + threadIdx.y;

    if(j < outer) {
        if(k < inter) {
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
        }
    }
}

__global__ void apply_blur_filter_loop_final(pixel* source, pixel* dest, int threshold, int end, int outer, int inter) {
    int j, k;

    j = blockIdx.x * blockDim.x + threadIdx.x;
    k = blockIdx.y * blockDim.y + threadIdx.y;

    if(j < outer) {
        if (k < inter) {
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
                end = 0;
            }

            source[CONV(j, k, width)].r = dest[CONV(j, k, width)].r;
            source[CONV(j, k, width)].g = dest[CONV(j, k, width)].g;
            source[CONV(j, k, width)].b = dest[CONV(j, k, width)].b;
        }
    }
}

void apply_blur_filter_cuda(int omprank, animated_gif *image, int size, int threshold, int image_index, striping_info* s_info) {
    int j, k;
    int width, height;
    int end = 0;
    int n_iter = 0;

    pixel *p;
    pixel *new;

    pixel *p_device;
    pixel *new_device;

    /* Get the pixels of all images */
    p = (image->p)[image_index];


    /* Process all images */
    n_iter = 0;
    width = image->width[image_index];
    height = image->height[image_index];

    /* Allocate array of new pixels */
    new = (pixel *) malloc(width * height * sizeof(pixel));

    checkCudaErrors(cudaMalloc((void**) &p_device, width * height * sizeof(pixel), streams[0]));
    checkCudaErrors(cudaMalloc((void**) &new_device, width * height * sizeof(pixel), streams[1]));

    checkCudaErrors(cudaMemcpy(p_device, (image->p)[image_index], width * height * sizeof(pixel),
                               cudaMemcpyHostToDevice, streams[0]));


    cudaMemcpy(C, dC, N * sizeof(double), cudaMemcpyDeviceToHost);

    do {
        end = 1;
        n_iter++;

        checkCudaErrors(cudaMemcpy(p, p_device, width * height * sizeof(pixel), cudaMemcpyDeviceToHost));
    } while(threshold > 0 && !end);

    /* Perform at least one blur iteration */

    do {
        end = 1;
        n_iter++;

        if (!s_info->single_mode) {
            synchronize_rows(p, width, height, size, s_info);
        }

#pragma omp parallel
        {
#pragma omp for collapse(2) private(j, k) firstprivate(new, p, width, height, size, s_info) schedule(static) nowait
            for (j = max(0, s_info->min_row); j < min(size, s_info->max_row); j++) {
                for (k = 0; k < width - 1; k++) {
                    new[CONV(j, k, width)].r = p[CONV(j, k, width)].r;
                    new[CONV(j, k, width)].g = p[CONV(j, k, width)].g;
                    new[CONV(j, k, width)].b = p[CONV(j, k, width)].b;
                }
            }

#pragma omp for collapse(2) private(j, k) firstprivate(new, p, width, height, size, s_info) schedule(static) nowait
            for (j = max(size, s_info->min_row); j < min(height - size, s_info->max_row); j++) {
                for (k = 0; k < size; k++) {
                    new[CONV(j, k, width)].r = p[CONV(j, k, width)].r;
                    new[CONV(j, k, width)].g = p[CONV(j, k, width)].g;
                    new[CONV(j, k, width)].b = p[CONV(j, k, width)].b;
                }
            }

#pragma omp for collapse(2) private(j, k) firstprivate(new, p, width, height, size, s_info) schedule(static) nowait
            for (j = max(size, s_info->min_row); j < min(height - size, s_info->max_row); j++) {
                for (k = width - size; k < width - 1; k++) {
                    new[CONV(j, k, width)].r = p[CONV(j, k, width)].r;
                    new[CONV(j, k, width)].g = p[CONV(j, k, width)].g;
                    new[CONV(j, k, width)].b = p[CONV(j, k, width)].b;
                }
            }

#pragma omp for collapse(2) private(j, k) firstprivate(new, p, width, height, size, s_info) schedule(static) nowait
            for (j = max(height - size, s_info->min_row); j < min(height - 1, s_info->max_row); j++) {
                for (k = 0; k < width - 1; k++) {
                    new[CONV(j, k, width)].r = p[CONV(j, k, width)].r;
                    new[CONV(j, k, width)].g = p[CONV(j, k, width)].g;
                    new[CONV(j, k, width)].b = p[CONV(j, k, width)].b;
                }
            }

            /* Apply blur on top part of image (10%) */
#pragma omp for collapse(2) private(j, k) firstprivate(p, new, width, height, s_info) schedule(static) nowait
            for (j = max(size, s_info->min_row); j < min(height / 10 - size, s_info->max_row); j++) {
                for (k = size; k < width - size; k++) {
                    int stencil_j, stencil_k;
                    int t_r = 0;
                    int t_g = 0;
                    int t_b = 0;

                    for (stencil_j = -size; stencil_j <= size; stencil_j++) {
                        for (stencil_k = -size; stencil_k <= size; stencil_k++) {
                            t_r += p[CONV(j + stencil_j, k + stencil_k, width)].r;
                            t_g += p[CONV(j + stencil_j, k + stencil_k, width)].g;
                            t_b += p[CONV(j + stencil_j, k + stencil_k, width)].b;
                        }
                    }

                    blur_filter_cuda(omp_get_thread_num(), size, p, j, k; width);

                    new[CONV(j, k, width)].r = t_r / ((2 * size + 1) * (2 * size + 1));
                    new[CONV(j, k, width)].g = t_g / ((2 * size + 1) * (2 * size + 1));
                    new[CONV(j, k, width)].b = t_b / ((2 * size + 1) * (2 * size + 1));
                }
            }

            /* Copy the middle part of the image */
#pragma omp for collapse(2) private(j, k) firstprivate(p, new, width, height, size, s_info) schedule(static) nowait
            for (j = max(height / 10 - size, s_info->min_row); j < min((int) (height * 0.9 + size), s_info->max_row);
                 j++) {
                for (k = size; k < width - size; k++) {
                    new[CONV(j, k, width)].r = p[CONV(j, k, width)].r;
                    new[CONV(j, k, width)].g = p[CONV(j, k, width)].g;
                    new[CONV(j, k, width)].b = p[CONV(j, k, width)].b;
                }
            }

            /* Apply blur on the bottom part of the image (10%) */
#pragma omp for collapse(2) private(j, k) firstprivate(width, height, new, p, size, s_info) schedule(static)
            for (j = max((int) (height * 0.9 + size), s_info->min_row); j < min(height - size, s_info->max_row); j++) {
                for (k = size; k < width - size; k++) {
                    int stencil_j, stencil_k;
                    int t_r = 0;
                    int t_g = 0;
                    int t_b = 0;

                    for (stencil_j = -size; stencil_j <= size; stencil_j++) {
                        for (stencil_k = -size; stencil_k <= size; stencil_k++) {
                            t_r += p[CONV(j + stencil_j, k + stencil_k, width)].r;
                            t_g += p[CONV(j + stencil_j, k + stencil_k, width)].g;
                            t_b += p[CONV(j + stencil_j, k + stencil_k, width)].b;
                        }
                    }

                    new[CONV(j, k, width)].r = t_r / ((2 * size + 1) * (2 * size + 1));
                    new[CONV(j, k, width)].g = t_g / ((2 * size + 1) * (2 * size + 1));
                    new[CONV(j, k, width)].b = t_b / ((2 * size + 1) * (2 * size + 1));
                }
            }

            // Sync point

#pragma omp for collapse(2) private(j, k) firstprivate(p, new, width, height, threshold, s_info) schedule(static)
            for (j = max(1, s_info->min_row); j < min(height - 1, s_info->max_row); j++) {
                for (k = 1; k < width - 1; k++) {

                    float diff_r;
                    float diff_g;
                    float diff_b;

                    diff_r = (new[CONV(j, k, width)].r - p[CONV(j, k, width)].r);
                    diff_g = (new[CONV(j, k, width)].g - p[CONV(j, k, width)].g);
                    diff_b = (new[CONV(j, k, width)].b - p[CONV(j, k, width)].b);

                    if (diff_r > threshold || -diff_r > threshold
                        ||
                        diff_g > threshold || -diff_g > threshold
                        ||
                        diff_b > threshold || -diff_b > threshold
                            ) {
                        end = 0;
                    }

                    p[CONV(j, k, width)].r = new[CONV(j, k, width)].r;
                    p[CONV(j, k, width)].g = new[CONV(j, k, width)].g;
                    p[CONV(j, k, width)].b = new[CONV(j, k, width)].b;
                }

            }

            if (!s_info->single_mode) {
                synchronize_bool_and(&end, s_info);
            }
        }
    } while (threshold > 0 && !end);
#if SOBELF_DEBUG
    printf( "BLUR: number of iterations for image %d\n", n_iter ) ;
#endif

    free(new);
}
