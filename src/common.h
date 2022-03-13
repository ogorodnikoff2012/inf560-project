//
// Created by xenon on 12.03.2022.
//

#ifndef SOBELF_COMMON_H
#define SOBELF_COMMON_H

#include "gif_lib.h"

/* Represent one pixel from the image */
typedef struct {
    uint8_t r; /* Red */
    uint8_t g; /* Green */
    uint8_t b; /* Blue */
} pixel;


/* Represent one GIF image (animated or not */
typedef struct {
    int n_images; /* Number of images */
    int *width; /* Width of each image */
    int *height; /* Height of each image */
    pixel **p; /* Pixels of each image */
    GifFileType *g; /* Internal representation.
                         DO NOT MODIFY */
} animated_gif;

typedef struct {
    int min_row;
    int max_row;
    int single_mode;
    int top_neighbour_id;
    int bottom_neighbour_id;
    int stripe_count;
} striping_info;

typedef struct {
    int n_images;
    int world_size;
    striping_info** s_info;
} collection_config;

typedef enum {
    TOP_TO_BTM_TAG,
    BTM_TO_TOP_TAG,
    SIGNAL_TAG,
    DATA_TAG,
} striping_mpi_tag;

#define CONV(l, c, nb_c) \
    ((l)*(nb_c)+(c))

#ifdef __cplusplus
extern "C" {
#endif

void allocate_device_MPI_process(int rank);

void apply_blur_filter_cuda(
        animated_gif* image,
        int size,
        int threshold,
        int image_index,
        striping_info* s_info);

void createCudaStreams();
void destroyCudaStreams();

#ifdef __cplusplus
}
#endif

#endif //SOBELF_COMMON_H
