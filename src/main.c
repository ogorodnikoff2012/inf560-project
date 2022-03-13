/*
 * INF560
 *
 * Image Filtering Project
 */
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h>

#include <mpi.h>

#include <unistd.h>
#include <limits.h>

#include "gif_lib.h"
#include "common.h"

/* Set this macro to 1 to enable debugging information */
#define SOBELF_DEBUG 0

MPI_Datatype kMPIPixelDatatype;
MPI_Datatype kMPIStripingInfoDatatype;

/*
 * Load a GIF image from a file and return a
 * structure of type animated_gif.
 */
animated_gif *load_pixels(char *filename) {
    GifFileType *g;
    ColorMapObject *colmap;
    int error;
    int n_images;
    int *width;
    int *height;
    pixel **p;
    int i;
    animated_gif *image;

    /* Open the GIF image (read mode) */
    g = DGifOpenFileName(filename, &error);

    if (g == NULL) {
        fprintf(stderr, "Error DGifOpenFileName %s\n", filename);
        return NULL;
    }

    /* Read the GIF image */
    error = DGifSlurp(g);

    if (error != GIF_OK) {
        fprintf(stderr,
                "Error DGifSlurp: %d <%s>\n", error, GifErrorString(g->Error));
        return NULL;
    }

    /* Grab the number of images and the size of each image */
    n_images = g->ImageCount;

    width = (int *) malloc(n_images * sizeof(int));

    if (width == NULL) {
        fprintf(stderr, "Unable to allocate width of size %d\n",
                n_images);
        return 0;
    }

    height = (int *) malloc(n_images * sizeof(int));

    if (height == NULL) {
        fprintf(stderr, "Unable to allocate height of size %d\n",
                n_images);
        return 0;
    }

    /* Fill the width and height */
    for (i = 0; i < n_images; i++) {
        width[i] = g->SavedImages[i].ImageDesc.Width;
        height[i] = g->SavedImages[i].ImageDesc.Height;

#if SOBELF_DEBUG
        printf( "Image %d: l:%d t:%d w:%d h:%d interlace:%d localCM:%p\n",
                i,
                g->SavedImages[i].ImageDesc.Left,
                g->SavedImages[i].ImageDesc.Top,
                g->SavedImages[i].ImageDesc.Width,
                g->SavedImages[i].ImageDesc.Height,
                g->SavedImages[i].ImageDesc.Interlace,
                g->SavedImages[i].ImageDesc.ColorMap
              ) ;
#endif
    }


    /* Get the global colormap */
    colmap = g->SColorMap;

    if (colmap == NULL) {
        fprintf(stderr, "Error global colormap is NULL\n");
        return NULL;
    }

#if SOBELF_DEBUG
    printf( "Global color map: count:%d bpp:%d sort:%d\n",
            g->SColorMap->ColorCount,
            g->SColorMap->BitsPerPixel,
            g->SColorMap->SortFlag
          ) ;
#endif

    /* Allocate the array of pixels to be returned */
    p = (pixel **) malloc(n_images * sizeof(pixel *));

    if (p == NULL) {
        fprintf(stderr, "Unable to allocate array of %d images\n",
                n_images);
        return NULL;
    }

    for (i = 0; i < n_images; i++) {
        p[i] = (pixel *) malloc(width[i] * height[i] * sizeof(pixel));

        if (p[i] == NULL) {
            fprintf(stderr, "Unable to allocate %d-th array of %d pixels\n",
                    i, width[i] * height[i]);
            return NULL;
        }
    }

    /* Fill pixels */

    /* For each image */
    for (i = 0; i < n_images; i++) {
        int j;

        /* Get the local colormap if needed */
        if (g->SavedImages[i].ImageDesc.ColorMap) {

            /* TODO No support for local color map */
            fprintf(stderr, "Error: application does not support local colormap\n");
            return NULL;

            colmap = g->SavedImages[i].ImageDesc.ColorMap;
        }

        /* Traverse the image and fill pixels */
        for (j = 0; j < width[i] * height[i]; j++) {
            int c;

            c = g->SavedImages[i].RasterBits[j];

            p[i][j].r = colmap->Colors[c].Red;
            p[i][j].g = colmap->Colors[c].Green;
            p[i][j].b = colmap->Colors[c].Blue;
        }
    }

    /* Allocate image info */
    image = (animated_gif *) malloc(sizeof(animated_gif));

    if (image == NULL) {
        fprintf(stderr, "Unable to allocate memory for animated_gif\n");
        return NULL;
    }

    /* Fill image fields */
    image->n_images = n_images;
    image->width = width;
    image->height = height;
    image->p = p;
    image->g = g;

#if SOBELF_DEBUG
    printf( "-> GIF w/ %d image(s) with first image of size %d x %d\n",
            image->n_images, image->width[0], image->height[0] ) ;
#endif

    return image;
}

int output_modified_read_gif(char *filename, GifFileType *g) {
    GifFileType *g2;
    int error2;

#if SOBELF_DEBUG
    printf( "Starting output to file %s\n", filename ) ;
#endif

    g2 = EGifOpenFileName(filename, false, &error2);

    if (g2 == NULL) {
        fprintf(stderr, "Error EGifOpenFileName %s\n",
                filename);
        return 0;
    }

    g2->SWidth = g->SWidth;
    g2->SHeight = g->SHeight;
    g2->SColorResolution = g->SColorResolution;
    g2->SBackGroundColor = g->SBackGroundColor;
    g2->AspectByte = g->AspectByte;
    g2->SColorMap = g->SColorMap;
    g2->ImageCount = g->ImageCount;
    g2->SavedImages = g->SavedImages;
    g2->ExtensionBlockCount = g->ExtensionBlockCount;
    g2->ExtensionBlocks = g->ExtensionBlocks;

    error2 = EGifSpew(g2);

    if (error2 != GIF_OK) {
        fprintf(stderr, "Error after writing g2: %d <%s>\n",
                error2, GifErrorString(g2->Error));
        return 0;
    }

    return 1;
}

int store_pixels(char *filename, animated_gif *image) {
    int n_colors = 0;
    pixel **p;
    int i, j, k;
    GifColorType *colormap;

    /* Initialize the new set of colors */
    colormap = (GifColorType *) malloc(256 * sizeof(GifColorType));

    if (colormap == NULL) {
        fprintf(stderr,
                "Unable to allocate 256 colors\n");
        return 0;
    }

    /* Everything is white by default */
    for (i = 0; i < 256; i++) {
        colormap[i].Red = 255;
        colormap[i].Green = 255;
        colormap[i].Blue = 255;
    }

    /* Change the background color and store it */
    int moy;
    moy = (
                  image->g->SColorMap->Colors[image->g->SBackGroundColor].Red
                  +
                  image->g->SColorMap->Colors[image->g->SBackGroundColor].Green
                  +
                  image->g->SColorMap->Colors[image->g->SBackGroundColor].Blue
          ) / 3;

    if (moy < 0) {
        moy = 0;
    }

    if (moy > 255) {
        moy = 255;
    }

#if SOBELF_DEBUG
    printf( "[DEBUG] Background color (%d,%d,%d) -> (%d,%d,%d)\n",
            image->g->SColorMap->Colors[ image->g->SBackGroundColor ].Red,
            image->g->SColorMap->Colors[ image->g->SBackGroundColor ].Green,
            image->g->SColorMap->Colors[ image->g->SBackGroundColor ].Blue,
            moy, moy, moy ) ;
#endif

    colormap[0].Red = moy;
    colormap[0].Green = moy;
    colormap[0].Blue = moy;

    image->g->SBackGroundColor = 0;

    n_colors++;

    /* Process extension blocks in main structure */
    for (j = 0; j < image->g->ExtensionBlockCount; j++) {
        int f;

        f = image->g->ExtensionBlocks[j].Function;

        if (f == GRAPHICS_EXT_FUNC_CODE) {
            int tr_color = image->g->ExtensionBlocks[j].Bytes[3];

            if (tr_color >= 0 &&
                tr_color < 255) {

                int found = -1;

                moy =
                        (
                                image->g->SColorMap->Colors[tr_color].Red
                                +
                                image->g->SColorMap->Colors[tr_color].Green
                                +
                                image->g->SColorMap->Colors[tr_color].Blue
                        ) / 3;

                if (moy < 0) {
                    moy = 0;
                }

                if (moy > 255) {
                    moy = 255;
                }

#if SOBELF_DEBUG
                printf( "[DEBUG] Transparency color image %d (%d,%d,%d) -> (%d,%d,%d)\n",
                        i,
                        image->g->SColorMap->Colors[ tr_color ].Red,
                        image->g->SColorMap->Colors[ tr_color ].Green,
                        image->g->SColorMap->Colors[ tr_color ].Blue,
                        moy, moy, moy ) ;
#endif

                for (k = 0; k < n_colors; k++) {
                    if (
                            moy == colormap[k].Red
                            &&
                            moy == colormap[k].Green
                            &&
                            moy == colormap[k].Blue
                            ) {
                        found = k;
                    }
                }

                if (found == -1) {
                    if (n_colors >= 256) {
                        fprintf(stderr,
                                "Error: Found too many colors inside the image\n"
                        );
                        return 0;
                    }

#if SOBELF_DEBUG
                    printf( "[DEBUG]\tNew color %d\n",
                            n_colors ) ;
#endif

                    colormap[n_colors].Red = moy;
                    colormap[n_colors].Green = moy;
                    colormap[n_colors].Blue = moy;


                    image->g->ExtensionBlocks[j].Bytes[3] = n_colors;

                    n_colors++;
                } else {
#if SOBELF_DEBUG
                    printf( "[DEBUG]\tFound existing color %d\n",
                            found ) ;
#endif
                    image->g->ExtensionBlocks[j].Bytes[3] = found;
                }
            }
        }
    }

    for (i = 0; i < image->n_images; i++) {
        for (j = 0; j < image->g->SavedImages[i].ExtensionBlockCount; j++) {
            int f;

            f = image->g->SavedImages[i].ExtensionBlocks[j].Function;

            if (f == GRAPHICS_EXT_FUNC_CODE) {
                int tr_color = image->g->SavedImages[i].ExtensionBlocks[j].Bytes[3];

                if (tr_color >= 0 &&
                    tr_color < 255) {

                    int found = -1;

                    moy =
                            (
                                    image->g->SColorMap->Colors[tr_color].Red
                                    +
                                    image->g->SColorMap->Colors[tr_color].Green
                                    +
                                    image->g->SColorMap->Colors[tr_color].Blue
                            ) / 3;

                    if (moy < 0) {
                        moy = 0;
                    }

                    if (moy > 255) {
                        moy = 255;
                    }

#if SOBELF_DEBUG
                    printf( "[DEBUG] Transparency color image %d (%d,%d,%d) -> (%d,%d,%d)\n",
                            i,
                            image->g->SColorMap->Colors[ tr_color ].Red,
                            image->g->SColorMap->Colors[ tr_color ].Green,
                            image->g->SColorMap->Colors[ tr_color ].Blue,
                            moy, moy, moy ) ;
#endif

                    for (k = 0; k < n_colors; k++) {
                        if (
                                moy == colormap[k].Red
                                &&
                                moy == colormap[k].Green
                                &&
                                moy == colormap[k].Blue
                                ) {
                            found = k;
                        }
                    }

                    if (found == -1) {
                        if (n_colors >= 256) {
                            fprintf(stderr,
                                    "Error: Found too many colors inside the image\n"
                            );
                            return 0;
                        }

#if SOBELF_DEBUG
                        printf( "[DEBUG]\tNew color %d\n",
                                n_colors ) ;
#endif

                        colormap[n_colors].Red = moy;
                        colormap[n_colors].Green = moy;
                        colormap[n_colors].Blue = moy;


                        image->g->SavedImages[i].ExtensionBlocks[j].Bytes[3] = n_colors;

                        n_colors++;
                    } else {
#if SOBELF_DEBUG
                        printf( "[DEBUG]\tFound existing color %d\n",
                                found ) ;
#endif
                        image->g->SavedImages[i].ExtensionBlocks[j].Bytes[3] = found;
                    }
                }
            }
        }
    }

#if SOBELF_DEBUG
    printf( "[DEBUG] Number of colors after background and transparency: %d\n",
            n_colors ) ;
#endif

    p = image->p;

    /* Find the number of colors inside the image */
    for (i = 0; i < image->n_images; i++) {

#if SOBELF_DEBUG
        printf( "OUTPUT: Processing image %d (total of %d images) -> %d x %d\n",
                i, image->n_images, image->width[i], image->height[i] ) ;
#endif

        for (j = 0; j < image->width[i] * image->height[i]; j++) {
            int found = 0;

            for (k = 0; k < n_colors; k++) {
                if (p[i][j].r == colormap[k].Red &&
                    p[i][j].g == colormap[k].Green &&
                    p[i][j].b == colormap[k].Blue) {
                    found = 1;
                }
            }

            if (found == 0) {
                if (n_colors >= 256) {
                    fprintf(stderr,
                            "Error: Found too many colors inside the image\n"
                    );
                    return 0;
                }

#if SOBELF_DEBUG
                printf( "[DEBUG] Found new %d color (%d,%d,%d)\n",
                        n_colors, p[i][j].r, p[i][j].g, p[i][j].b ) ;
#endif

                colormap[n_colors].Red = p[i][j].r;
                colormap[n_colors].Green = p[i][j].g;
                colormap[n_colors].Blue = p[i][j].b;
                n_colors++;
            }
        }
    }

#if SOBELF_DEBUG
    printf( "OUTPUT: found %d color(s)\n", n_colors ) ;
#endif


    /* Round up to a power of 2 */
    if (n_colors != (1 << GifBitSize(n_colors))) {
        n_colors = (1 << GifBitSize(n_colors));
    }

#if SOBELF_DEBUG
    printf( "OUTPUT: Rounding up to %d color(s)\n", n_colors ) ;
#endif

    /* Change the color map inside the animated gif */
    ColorMapObject *cmo;

    cmo = GifMakeMapObject(n_colors, colormap);

    if (cmo == NULL) {
        fprintf(stderr, "Error while creating a ColorMapObject w/ %d color(s)\n",
                n_colors);
        return 0;
    }

    image->g->SColorMap = cmo;

    /* Update the raster bits according to color map */
    for (i = 0; i < image->n_images; i++) {
        for (j = 0; j < image->width[i] * image->height[i]; j++) {
            int found_index = -1;

            for (k = 0; k < n_colors; k++) {
                if (p[i][j].r == image->g->SColorMap->Colors[k].Red &&
                    p[i][j].g == image->g->SColorMap->Colors[k].Green &&
                    p[i][j].b == image->g->SColorMap->Colors[k].Blue) {
                    found_index = k;
                }
            }

            if (found_index == -1) {
                fprintf(stderr,
                        "Error: Unable to find a pixel in the color map\n");
                return 0;
            }

            image->g->SavedImages[i].RasterBits[j] = found_index;
        }
    }


    /* Write the final image */
    if (!output_modified_read_gif(filename, image->g)) {
        return 0;
    }

    return 1;
}

void apply_gray_filter(animated_gif *image, int image_index, striping_info* s_info) {
    int row, col;
    pixel *p;

    p = (image->p)[image_index];

    const int width = image->width[image_index];

    const int min_row = s_info->min_row;
    const int max_row = s_info->max_row;

#pragma omp parallel for collapse(2) schedule(static) firstprivate(p, width, min_row, max_row) private(row, col)
    for (row = min_row; row < max_row; row++) {
        for (col = 0; col < width; ++col) {
            int moy;

            moy = (p[CONV(row, col, width)].r + p[CONV(row, col, width)].g + p[CONV(row, col, width)].b) / 3;

            if (moy < 0) {
                moy = 0;
            }

            if (moy > 255) {
                moy = 255;
            }

            p[CONV(row, col, width)].r = moy;
            p[CONV(row, col, width)].g = moy;
            p[CONV(row, col, width)].b = moy;
        }
    }
}

void apply_gray_line(animated_gif *image) {
    int i, j, k;
    pixel **p;

    p = image->p;

    for (i = 0; i < image->n_images; i++) {
        for (j = 0; j < 10; j++) {
            for (k = image->width[i] / 2; k < image->width[i]; k++) {
                p[i][CONV(j, k, image->width[i])].r = 0;
                p[i][CONV(j, k, image->width[i])].g = 0;
                p[i][CONV(j, k, image->width[i])].b = 0;
            }
        }
    }
}

void synchronize_rows(pixel* p, int width, int height, int row_count, striping_info* s_info) {
#define TOP_RECV 0
#define TOP_SEND 1
#define BTM_RECV 2
#define BTM_SEND 3

    MPI_Request requests[] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL};

    if (s_info->top_neighbour_id != -1) {
        MPI_Isend(p + CONV(s_info->min_row, 0, width), width * row_count, kMPIPixelDatatype,
                  s_info->top_neighbour_id, TOP_TO_BTM_TAG, MPI_COMM_WORLD, &requests[TOP_SEND]);
        MPI_Irecv(p + CONV(s_info->min_row - row_count, 0, width), width * row_count, kMPIPixelDatatype,
                  s_info->top_neighbour_id, BTM_TO_TOP_TAG, MPI_COMM_WORLD, &requests[TOP_RECV]);
    }

    if (s_info->bottom_neighbour_id != -1) {
        MPI_Isend(p + CONV(s_info->max_row - row_count, 0, width), width * row_count, kMPIPixelDatatype,
                  s_info->bottom_neighbour_id, BTM_TO_TOP_TAG, MPI_COMM_WORLD, &requests[BTM_SEND]);
        MPI_Irecv(p + CONV(s_info->max_row, 0, width), width * row_count, kMPIPixelDatatype,
                  s_info->bottom_neighbour_id, TOP_TO_BTM_TAG, MPI_COMM_WORLD, &requests[BTM_RECV]);
    }

    MPI_Waitall(sizeof(requests)/sizeof(requests[0]), requests, MPI_STATUSES_IGNORE);

#undef TOP_RECV
#undef TOP_SEND
#undef BTM_RECV
#undef BTM_SEND
}

void synchronize_bool_and(int* var, striping_info* s_info) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        for (int slave = 1; slave < s_info->stripe_count; ++slave) {
            int value;
            MPI_Recv(&value, 1, MPI_INT, slave, SIGNAL_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            *var &= value;
        }

        for (int slave = 1; slave < s_info->stripe_count; ++slave) {
            MPI_Send(var, 1, MPI_INT, slave, SIGNAL_TAG, MPI_COMM_WORLD);
        }
    } else {
        MPI_Send(var, 1, MPI_INT, 0, SIGNAL_TAG, MPI_COMM_WORLD);
        MPI_Recv(var, 1, MPI_INT, 0, SIGNAL_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}

static inline int min(int x, int y) {
    return x < y ? x : y;
}

static inline int max(int x, int y) {
    return x < y ? y : x;
}

void apply_blur_filter(animated_gif *image, int size, int threshold, int image_index, striping_info* s_info) {
    int j, k;
    int width, height;
    int end = 0;
    int n_iter = 0;

    pixel *p;
    pixel *new;

    /* Get the pixels of all images */
    p = (image->p)[image_index];


    /* Process all images */
    n_iter = 0;
    width = image->width[image_index];
    height = image->height[image_index];

    /* Allocate array of new pixels */
    new = (pixel *) malloc(width * height * sizeof(pixel));


    /* Perform at least one blur iteration */

    do {
        end = 1;
        n_iter++;

        if (!s_info->single_mode) {
            synchronize_rows(p, width, height, size, s_info);
        }

        const int begin1    = max(0, s_info->min_row);
        const int end1      = min(size, s_info->max_row);
        const int begin2    = max(size, s_info->min_row);
        const int end2      = min(height - size, s_info->max_row);
        const int begin3    = max(size, s_info->min_row);
        const int end3      = min(height - size, s_info->max_row);
        const int begin4    = max(height - size, s_info->min_row);
        const int end4      = min(height - 1, s_info->max_row);
        const int begin5    = max(size, s_info->min_row);
        const int end5      = min(height / 10 - size, s_info->max_row);
        const int begin6    = max(height / 10 - size, s_info->min_row);
        const int end6      = min((int)(height * 0.9 + size), s_info->max_row);
        const int begin7    = max((int)(height * 0.9 + size), s_info->min_row);
        const int end7      = min(height - size, s_info->max_row);
        const int begin8    = max(1, s_info->min_row);
        const int end8      = min(height - 1, s_info->max_row);

        #pragma omp parallel shared(end)
        {
#pragma omp for collapse(2) private(j, k) firstprivate(new, p, width, height, size, s_info) schedule(static) nowait
            for (j = begin1; j < end1; j++) {
                for (k = 0; k < width - 1; k++) {
                    new[CONV(j, k, width)].r = p[CONV(j, k, width)].r;
                    new[CONV(j, k, width)].g = p[CONV(j, k, width)].g;
                    new[CONV(j, k, width)].b = p[CONV(j, k, width)].b;
                }
            }

#pragma omp for collapse(2) private(j, k) firstprivate(new, p, width, height, size, s_info) schedule(static) nowait
            for (j = begin2; j < end2; j++) {
                for (k = 0; k < size; k++) {
                    new[CONV(j, k, width)].r = p[CONV(j, k, width)].r;
                    new[CONV(j, k, width)].g = p[CONV(j, k, width)].g;
                    new[CONV(j, k, width)].b = p[CONV(j, k, width)].b;
                }
            }

#pragma omp for collapse(2) private(j, k) firstprivate(new, p, width, height, size, s_info) schedule(static) nowait
            for (j = begin3; j < end3; j++) {
                for (k = width - size; k < width - 1; k++) {
                    new[CONV(j, k, width)].r = p[CONV(j, k, width)].r;
                    new[CONV(j, k, width)].g = p[CONV(j, k, width)].g;
                    new[CONV(j, k, width)].b = p[CONV(j, k, width)].b;
                }
            }

#pragma omp for collapse(2) private(j, k) firstprivate(new, p, width, height, size, s_info) schedule(static) nowait
            for (j = begin4; j < end4; j++) {
                for (k = 0; k < width - 1; k++) {
                    new[CONV(j, k, width)].r = p[CONV(j, k, width)].r;
                    new[CONV(j, k, width)].g = p[CONV(j, k, width)].g;
                    new[CONV(j, k, width)].b = p[CONV(j, k, width)].b;
                }
            }

            /* Apply blur on top part of image (10%) */
#pragma omp for collapse(2) private(j, k) firstprivate(p, new, width, height, s_info) schedule(static) nowait
            for (j = begin5; j < end5; j++) {
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

            /* Copy the middle part of the image */
#pragma omp for collapse(2) private(j, k) firstprivate(p, new, width, height, size, s_info) schedule(static) nowait
            for (j = begin6; j < end6; j++) {
                for (k = size; k < width - size; k++) {
                    new[CONV(j, k, width)].r = p[CONV(j, k, width)].r;
                    new[CONV(j, k, width)].g = p[CONV(j, k, width)].g;
                    new[CONV(j, k, width)].b = p[CONV(j, k, width)].b;
                }
            }

            /* Apply blur on the bottom part of the image (10%) */
#pragma omp for collapse(2) private(j, k) firstprivate(width, height, new, p, size, s_info) schedule(static)
            for (j = begin7; j < end7; j++) {
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

#pragma omp for collapse(2) private(j, k) firstprivate(p, new, width, height, threshold, s_info) schedule(static) reduction(&&: end)
            for (j = begin8; j < end8; j++) {
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
                        end = end && 0;
                    }

                    p[CONV(j, k, width)].r = new[CONV(j, k, width)].r;
                    p[CONV(j, k, width)].g = new[CONV(j, k, width)].g;
                    p[CONV(j, k, width)].b = new[CONV(j, k, width)].b;
                }
            }

        }

        if (!s_info->single_mode) {
            synchronize_bool_and(&end, s_info);
        }
    } while (threshold > 0 && !end);
#if SOBELF_DEBUG
    printf( "BLUR: number of iterations for image %d\n", n_iter ) ;
#endif

    free(new);

}

void apply_sobel_filter(animated_gif *image, int image_index, striping_info* s_info) {
    int j, k;
    int width, height;

    pixel *p;

    p = (image->p)[image_index];

    width = image->width[image_index];
    height = image->height[image_index];

    pixel *sobel;

    sobel = (pixel *) malloc(width * height * sizeof(pixel));

    if (!s_info->single_mode) {
        synchronize_rows(p, width, height, 1, s_info);
    }

    #pragma omp parallel
    {
        const int beginOuter = max(1, s_info->min_row);
        const int endOuter   = min(height - 1, s_info->max_row);

#pragma omp for collapse(2) private(j, k) firstprivate(p, sobel, width, height, s_info) schedule(static)
        for (j = beginOuter; j < endOuter; j++) {
            for (k = 1; k < width - 1; k++) {
                int pixel_blue_no, pixel_blue_n, pixel_blue_ne;
                int pixel_blue_so, pixel_blue_s, pixel_blue_se;
                int pixel_blue_o, pixel_blue, pixel_blue_e;

                float deltaX_blue;
                float deltaY_blue;
                float val_blue;

                pixel_blue_no = p[CONV(j - 1, k - 1, width)].b;
                pixel_blue_n = p[CONV(j - 1, k, width)].b;
                pixel_blue_ne = p[CONV(j - 1, k + 1, width)].b;
                pixel_blue_so = p[CONV(j + 1, k - 1, width)].b;
                pixel_blue_s = p[CONV(j + 1, k, width)].b;
                pixel_blue_se = p[CONV(j + 1, k + 1, width)].b;
                pixel_blue_o = p[CONV(j, k - 1, width)].b;
                pixel_blue = p[CONV(j, k, width)].b;
                pixel_blue_e = p[CONV(j, k + 1, width)].b;

                deltaX_blue = -pixel_blue_no + pixel_blue_ne - 2 * pixel_blue_o + 2 * pixel_blue_e - pixel_blue_so +
                              pixel_blue_se;

                deltaY_blue = pixel_blue_se + 2 * pixel_blue_s + pixel_blue_so - pixel_blue_ne - 2 * pixel_blue_n -
                              pixel_blue_no;

                val_blue = sqrt(deltaX_blue * deltaX_blue + deltaY_blue * deltaY_blue) / 4;


                if (val_blue > 50) {
                    sobel[CONV(j, k, width)].r = 255;
                    sobel[CONV(j, k, width)].g = 255;
                    sobel[CONV(j, k, width)].b = 255;
                } else {
                    sobel[CONV(j, k, width)].r = 0;
                    sobel[CONV(j, k, width)].g = 0;
                    sobel[CONV(j, k, width)].b = 0;
                }
            }
        }

#pragma omp for collapse(2) private(j, k) firstprivate(p, sobel, width, height, s_info) schedule(static)
        for (j = beginOuter; j < endOuter; j++) {
            for (k = 1; k < width - 1; k++) {
                p[CONV(j, k, width)].r = sobel[CONV(j, k, width)].r;
                p[CONV(j, k, width)].g = sobel[CONV(j, k, width)].g;
                p[CONV(j, k, width)].b = sobel[CONV(j, k, width)].b;
            }
        }
    }
    free(sobel);
}

#define BLUR_RADIUS (5)

void apply_all_filters(animated_gif *image, int image_idx, striping_info* s_info) {
    // Convert the pixels into grayscale
    apply_gray_filter(image, image_idx, s_info);

    // Apply blur filter with convergence value
#ifdef USE_CUDA
    if (s_info->single_mode) {
        apply_blur_filter_cuda(image, BLUR_RADIUS, 20, image_idx, s_info);
    } else {
        apply_blur_filter(image, BLUR_RADIUS, 20, image_idx, s_info);
    }
#else
    apply_blur_filter(image, BLUR_RADIUS, 20, image_idx, s_info);
#endif

    // Apply sobel filter on pixels
    apply_sobel_filter(image, image_idx, s_info);
}

void prepare_pixel_datatype(MPI_Datatype *datatype) {
    const int nitems = 3;
    int blocklengths[3] = {1, 1, 1};
    MPI_Datatype types[3] = {MPI_UINT8_T, MPI_UINT8_T, MPI_UINT8_T};
    MPI_Aint offsets[3];

    offsets[0] = offsetof(pixel, r);
    offsets[1] = offsetof(pixel, g);
    offsets[2] = offsetof(pixel, b);
    MPI_Type_create_struct(nitems, blocklengths, offsets, types, datatype);
    MPI_Type_commit(datatype);
}

void prapare_striping_info_datatype(MPI_Datatype* datatype) {
    const int nitems = 6;
    int blocklengths[6] = {1, 1, 1, 1, 1, 1};
    MPI_Datatype types[6] = {MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT};
    MPI_Aint offsets[6];

    offsets[0] = offsetof(striping_info, min_row);
    offsets[1] = offsetof(striping_info, max_row);
    offsets[2] = offsetof(striping_info, single_mode);
    offsets[3] = offsetof(striping_info, top_neighbour_id);
    offsets[4] = offsetof(striping_info, bottom_neighbour_id);
    offsets[5] = offsetof(striping_info, stripe_count);

    MPI_Type_create_struct(nitems, blocklengths, offsets, types, datatype);
    MPI_Type_commit(datatype);
}

void prepare_datatypes() {
    prepare_pixel_datatype(&kMPIPixelDatatype);
    prapare_striping_info_datatype(&kMPIStripingInfoDatatype);
}

int slave_sync(void) {
    int value;
    MPI_Bcast(&value, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return value;
}

void master_sync(int value) {
    MPI_Bcast(&value, 1, MPI_INT, 0, MPI_COMM_WORLD);
}

void send_signal_to_master() {
    int buffer = 1;
    MPI_Send(&buffer, 1, MPI_INT, 0, SIGNAL_TAG, MPI_COMM_WORLD);
}

void get_signal_from_slave(int slave_rank) {
    int buffer;
    MPI_Recv(&buffer, 1, MPI_INT, slave_rank, SIGNAL_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

#define WORK_MODE_FAILURE  (0)
#define WORK_MODE_LEGACY   (1)
#define WORK_MODE_STRIPING (2)

void slave_broadcast_metadata(animated_gif* image) {
    MPI_Bcast(&image->n_images, 1, MPI_INT, 0, MPI_COMM_WORLD);

    image->width = calloc(image->n_images, sizeof(int));
    image->height = calloc(image->n_images, sizeof(int));
    image->p = calloc(image->n_images, sizeof(pixel *));

    MPI_Bcast(image->width, image->n_images, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(image->height, image->n_images, MPI_INT, 0, MPI_COMM_WORLD);
}

void master_broadcast_metadata(animated_gif* image) {
    MPI_Bcast(&image->n_images, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(image->width, image->n_images, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(image->height, image->n_images, MPI_INT, 0, MPI_COMM_WORLD);
}

int slave_receive_stripe_info(striping_info* s_info) {
    MPI_Recv(s_info, 1, kMPIStripingInfoDatatype, 0, SIGNAL_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    int used = s_info->max_row >= 0 ? 1 : 0;

    /*
    printf("SLAVE  ok?%d s_i{min_r=%d,max_r=%d,s_m=%d,t_n=%d,b_n=%d,s_c=%d}\n",
           used,
           s_info->min_row,
           s_info->max_row,
           s_info->single_mode,
           s_info->top_neighbour_id,
           s_info->bottom_neighbour_id,
           s_info->stripe_count
    );
     */

    return used;
}

void slave_receive_stripe(pixel* p, int width, int height, striping_info* s_info) {
    int row_count = s_info->max_row - s_info->min_row;
    MPI_Recv(p + CONV(s_info->min_row, 0, width), width * row_count, kMPIPixelDatatype,
             0, DATA_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

void slave_send_stripe(pixel* p, int width, int height, striping_info* s_info) {
    int row_count = s_info->max_row - s_info->min_row;
    MPI_Send(p + CONV(s_info->min_row, 0, width), width * row_count, kMPIPixelDatatype,
             0, DATA_TAG, MPI_COMM_WORLD);
}

void clear_outer_pixels(pixel* p, int width, int height, striping_info* s_info) {
    int row, col;
    if (s_info->max_row == -1) {
#pragma omp parallel for collapse(2) schedule(static) firstprivate(p, width, height) private(row, col)
        for (row = 0; row < height; ++row) {
            for (col = 0; col < width; ++col) {
                p[CONV(row, col, width)].r = 0;
                p[CONV(row, col, width)].g = 0;
                p[CONV(row, col, width)].b = 0;
            }
        }
    } else {
#pragma omp parallel
        {
#pragma omp for collapse(2) schedule(static) firstprivate(p, width, height, s_info) private(row, col) nowait
            for (row = 0; row < s_info->min_row; ++row) {
                for (col = 0; col < width; ++col) {
                    p[CONV(row, col, width)].r = 0;
                    p[CONV(row, col, width)].g = 0;
                    p[CONV(row, col, width)].b = 0;
                }
            }
#pragma omp for collapse(2) schedule(static) firstprivate(p, width, height, s_info) private(row, col)
            for (row = s_info->max_row; row < height; ++row) {
                for (col = 0; col < width; ++col) {
                    p[CONV(row, col, width)].r = 0;
                    p[CONV(row, col, width)].g = 0;
                    p[CONV(row, col, width)].b = 0;
                }
            }
        }
    }
}

void copy_stripe(pixel* src, pixel* dst, int width, int height, striping_info* s_info) {
    if (s_info->max_row == -1) { return; }
    int row, col;
#pragma omp parallel for collapse(2) schedule(static) firstprivate(src, dst, width,height, s_info) private(row, col)
    for (row = s_info->min_row; row < s_info->max_row; ++row) {
        for (col = 0; col < width; ++col) {
            dst[CONV(row, col, width)].r = src[CONV(row, col, width)].r;
            dst[CONV(row, col, width)].g = src[CONV(row, col, width)].g;
            dst[CONV(row, col, width)].b = src[CONV(row, col, width)].b;
        }
    }
}

int slave_striping(animated_gif* image, char* output_filename) {
    // slave_broadcast_metadata(image);

    for (int image_idx = 0; image_idx < image->n_images; ++image_idx) {
        int width = image->width[image_idx];
        int height = image->height[image_idx];

        striping_info s_info;
        int has_work = slave_receive_stripe_info(&s_info);
        if (!has_work) {
            pixel* p = image->p[image_idx]; // = calloc(image->width[image_idx] * image->height[image_idx], sizeof(pixel));
            clear_outer_pixels(p, width, height, &s_info);
            continue;
        }

        pixel* p = image->p[image_idx]; // = calloc(image->width[image_idx] * image->height[image_idx], sizeof(pixel));
        // slave_receive_stripe(p, width, height, &s_info);
        apply_all_filters(image, image_idx, &s_info);
        clear_outer_pixels(p, width, height, &s_info);
        // slave_send_stripe(p, width, height, &s_info);
        // free(p);
    }

    // free(image->width);
    // free(image->height);
    // free(image->p);
    store_pixels(output_filename, image);
    send_signal_to_master();

    return 0;
}

char* generate_output_filename(const char* original, int rank) {
    int orig_len = strlen(original);
    int buf_len = orig_len + 100;
    char* buffer = calloc(buf_len, sizeof(char));
    snprintf(buffer, buf_len, "%s.%d.gif", original, rank);
    return buffer;
}

int slave_main(int argc, char *argv[]) {
    int rank;
    int world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int work_mode = slave_sync();

    if (work_mode == WORK_MODE_FAILURE) {
        return 1;
    }

    char* input_filename = argv[1];
    char* output_filename = generate_output_filename(argv[2], rank);

    animated_gif* image = load_pixels(input_filename);

    if (work_mode == WORK_MODE_STRIPING) {
        return slave_striping(image, output_filename);
    }

    if (work_mode != WORK_MODE_LEGACY) {
        printf("PANIC!!! Unknown work mode (%d)\n", work_mode);
        return 2;
    }

    /* First, we broadcast metadata */
    // slave_broadcast_metadata(&image);

    const int kSignalTag = image->n_images;

    int image_index = -1;
    MPI_Send(&image_index, 1, MPI_INT, 0, kSignalTag, MPI_COMM_WORLD);

    MPI_Request processed_image_requests[image->n_images];
    for (int i = 0; i < image->n_images; ++i) {
        processed_image_requests[i] = MPI_REQUEST_NULL;
    }

    bool processed_images[image->n_images];
    for (int i = 0; i < image->n_images; ++i) {
        processed_images[i] = false;
    }

    while (true) {
        MPI_Recv(&image_index, 1, MPI_INT, 0, kSignalTag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (image_index == -1) {
            break;
        }

        /*
        image.p[image_index] = calloc(image.width[image_index] * image.height[image_index], sizeof(pixel));
        MPI_Recv(image.p[image_index], image.width[image_index] * image.height[image_index], kMPIPixelDatatype, 0,
                 image_index, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        */

        processed_images[image_index] = true;

        striping_info s_info;
        s_info.single_mode = 1;
        s_info.min_row = 0;
        s_info.max_row = image->height[image_index];
        s_info.top_neighbour_id = -1;
        s_info.bottom_neighbour_id = -1;
        s_info.stripe_count = 1;

        apply_all_filters(image, image_index, &s_info);

        MPI_Request req;
        MPI_Isend(&image_index, 1, MPI_INT, 0, kSignalTag, MPI_COMM_WORLD, &req);
        MPI_Request_free(&req);

        // MPI_Isend(image->p[image_index], image->width[image_index] * image->height[image_index],
        // kMPIPixelDatatype, 0,
        //          image_index, MPI_COMM_WORLD, processed_image_requests + image_index);
    }

    for (int i = 0; i < image->n_images; ++i) {
        if (processed_image_requests[i] != MPI_REQUEST_NULL) {
            MPI_Wait(processed_image_requests + i, MPI_STATUS_IGNORE);
        }
        if (!processed_images[i]) {
            striping_info s_info;
            s_info.min_row = -1;
            s_info.max_row = -1;
            s_info.single_mode = 1;
            s_info.top_neighbour_id = -1;
            s_info.bottom_neighbour_id = -1;
            s_info.stripe_count = 1;
            clear_outer_pixels(image->p[i], image->width[i], image->height[i], &s_info);
        }
        // free(image->p[i]);
    }

    // free(image->width);
    // free(image->height);
    // free(image->p);
    store_pixels(output_filename, image);
    send_signal_to_master();

    return 0;
}

collection_config* do_master_work_legacy(animated_gif *image) {
    int rank;
    int world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    const int kSignalTag = image->n_images;

    printf("Working mode: legacy\n");

    /* First, we broadcast metadata */
    // master_broadcast_metadata(image);

    collection_config* cfg = calloc(1, sizeof(collection_config));
    cfg->n_images = image->n_images;
    cfg->world_size = world_size;
    cfg->s_info = calloc(cfg->n_images, sizeof(void*));

    /* Start scheduling */
    MPI_Request table_of_requests[world_size];
    int slave_signals[world_size];
    table_of_requests[0] = MPI_REQUEST_NULL;

    bool slave_terminated[world_size];
    for (int i = 0; i < world_size; ++i) {
        slave_terminated[i] = false;
    }

    MPI_Request processed_image_requests[image->n_images];
    int processed_images = 0;
    int sent_images = 0;
    for (int i = 0; i < image->n_images; ++i) {
        processed_image_requests[i] = MPI_REQUEST_NULL;
    }

    for (int i = 1; i < world_size; ++i) {
        MPI_Irecv(slave_signals + i, 1, MPI_INT, i, kSignalTag, MPI_COMM_WORLD, table_of_requests + i);
    }

    while (processed_images < image->n_images) {
        int indx;
        MPI_Waitany(world_size, table_of_requests, &indx, MPI_STATUS_IGNORE);

        int image_index = slave_signals[indx];
        if (image_index != -1) {
            striping_info* s_info = cfg->s_info[image_index] = calloc(world_size, sizeof(striping_info));
            for (int i = 0; i < world_size; ++i) {
                s_info[i].min_row = -1;
                s_info[i].max_row = -1;
                s_info[i].single_mode = 1;
                s_info[i].top_neighbour_id = -1;
                s_info[i].bottom_neighbour_id = -1;
                s_info[i].stripe_count = 1;
            }
            s_info[indx].min_row = 0;
            s_info[indx].max_row = image->height[image_index];

            // MPI_Irecv(image->p[image_index], image->width[image_index] * image->height[image_index],
            //           kMPIPixelDatatype, indx, image_index, MPI_COMM_WORLD, &processed_image_requests[image_index]);
            ++processed_images;
        }

        if (sent_images == image->n_images) {
            MPI_Request req;
            int next_image_index = -1;
            MPI_Isend(&next_image_index, 1, MPI_INT, indx, kSignalTag, MPI_COMM_WORLD, &req);
            MPI_Request_free(&req);
            slave_terminated[indx] = true;
        } else {
            MPI_Request req;
            int next_image_index = sent_images++;
            MPI_Isend(&next_image_index, 1, MPI_INT, indx, kSignalTag, MPI_COMM_WORLD, &req);
            MPI_Request_free(&req);
            // MPI_Isend(image->p[next_image_index], image->width[next_image_index] * image->height[next_image_index],
            //          kMPIPixelDatatype, indx, next_image_index, MPI_COMM_WORLD, &req);
            // MPI_Request_free(&req);

            MPI_Irecv(slave_signals + indx, 1, MPI_INT, indx, kSignalTag, MPI_COMM_WORLD, table_of_requests + indx);
        }
    }

    /* Finally, we wait for all images */
    for (int i = 0; i < image->n_images; ++i) {
        if (processed_image_requests[i] != MPI_REQUEST_NULL) {
            MPI_Wait(processed_image_requests + i, MPI_STATUS_IGNORE);
        }
    }

    /* And terminate all slaves */
    for (int i = 1; i < world_size; ++i) {
        if (!slave_terminated[i]) {
            MPI_Wait(table_of_requests + i, MPI_STATUS_IGNORE);

            MPI_Request req;
            int next_image_index = -1;
            MPI_Isend(&next_image_index, 1, MPI_INT, i, kSignalTag, MPI_COMM_WORLD, &req);
            MPI_Request_free(&req);
            slave_terminated[i] = true;
        }
    }

    return cfg;
}

int prepare_stripe_info(int height, int world_size, striping_info* s_info) {
    for (int i = 0; i < world_size; ++i) {
        s_info[i].max_row = -1;
    }

    int last_max_row = 0;
    int stripe_count = 0;
    while (last_max_row < height) {
        int stripe_height = max((height + world_size - 1) / world_size, BLUR_RADIUS);

        int min_row = last_max_row;
        int max_row = min_row + stripe_height;
        if (height - max_row < BLUR_RADIUS) {
            max_row = height;
        }
        if (max_row > height) {
            max_row = height;
        }

        s_info[stripe_count].min_row = min_row;
        s_info[stripe_count].max_row = max_row;
        s_info[stripe_count].single_mode = false;
        s_info[stripe_count].top_neighbour_id = -1;
        s_info[stripe_count].bottom_neighbour_id = -1;

        ++stripe_count;
        last_max_row = max_row;
    }

    for (int i = 0; i < stripe_count; ++i) {
        s_info[i].stripe_count = stripe_count;
        if (i > 0) {
            s_info[i].top_neighbour_id = i - 1;
        }
        if (i < stripe_count - 1) {
            s_info[i].bottom_neighbour_id = i + 1;
        }
    }

    return stripe_count;
}

void master_send_stripe(int slave_rank, pixel* p, int width, int height, striping_info* s_info, MPI_Request* requests) {
    int used = s_info->max_row >= 0 ? 1 : 0;
    /*
    printf("MASTER ok?%d s_i{min_r=%d,max_r=%d,s_m=%d,t_n=%d,b_n=%d,s_c=%d}\n",
        used,
        s_info->min_row,
        s_info->max_row,
        s_info->single_mode,
        s_info->top_neighbour_id,
        s_info->bottom_neighbour_id,
        s_info->stripe_count
    );
     */

    MPI_Isend(s_info, 1, kMPIStripingInfoDatatype, slave_rank, SIGNAL_TAG, MPI_COMM_WORLD, &requests[2 * slave_rank]);

    //if (!used) {
    requests[2 * slave_rank + 1] = MPI_REQUEST_NULL;
    return;
    //}

    //int row_count = s_info->max_row - s_info->min_row;
    //MPI_Isend(p + CONV(s_info->min_row, 0, width), width * row_count, kMPIPixelDatatype,
    //          slave_rank, DATA_TAG, MPI_COMM_WORLD, &requests[2 * slave_rank + 1]);
}

void master_receive_stripes(pixel* p, int width, int height, striping_info* s_info, int stripe_count) {
    MPI_Request requests[stripe_count];
    requests[0] = MPI_REQUEST_NULL;

    for (int i = 1; i < stripe_count; ++i) {
        int row_count = s_info[i].max_row - s_info[i].min_row;
        MPI_Irecv(p + CONV(s_info[i].min_row, 0, width), width * row_count, kMPIPixelDatatype,
                  i, DATA_TAG, MPI_COMM_WORLD, &requests[i]);
    }

    MPI_Waitall(stripe_count, requests, MPI_STATUSES_IGNORE);
}

collection_config* do_master_work_striping(animated_gif* image) {
    int rank;
    int world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    printf("Working mode: striping\n");

    /* First, we broadcast metadata */
    // master_broadcast_metadata(image);

    collection_config* cfg = calloc(1, sizeof(collection_config));
    cfg->n_images = image->n_images;
    cfg->world_size = world_size;
    cfg->s_info = calloc(cfg->n_images, sizeof(void*));

    for (int image_idx = 0; image_idx < image->n_images; ++image_idx) {
        int width = image->width[image_idx];
        int height = image->height[image_idx];

        striping_info* s_info = cfg->s_info[image_idx] = calloc(world_size, sizeof(striping_info));
        int stripe_count = prepare_stripe_info(height, world_size, s_info);

        // Send stripes

        MPI_Request requests[2 * world_size];
        requests[0] = requests[1] = MPI_REQUEST_NULL;

        for (int i = 1; i < world_size; ++i) {
            master_send_stripe(i, image->p[image_idx], width, height, &s_info[i], requests);
        }

        apply_all_filters(image, image_idx, &s_info[0]);

        MPI_Waitall(2 * world_size, requests, MPI_STATUSES_IGNORE);

        // master_receive_stripes(image->p[image_idx], width, height, s_info, stripe_count);
    }

    return cfg;
}

void collect_data(animated_gif* image, collection_config* cfg, const char* output_filename) {
//    // Just print it
//    for (int i = 0; i < cfg->n_images; ++i) {
//        printf("Image %3d/%3d\n", i, cfg->n_images);
//        for (int j = 0; j < cfg->world_size; ++j) {
//            striping_info* s_info = &cfg->s_info[i][j];
//            printf("SLAVE %2d/%2d s_i{min_r=%d,max_r=%d,s_m=%d,t_n=%d,b_n=%d,s_c=%d}\n",
//                   j, cfg->world_size,
//                   s_info->min_row,
//                   s_info->max_row,
//                   s_info->single_mode,
//                   s_info->top_neighbour_id,
//                   s_info->bottom_neighbour_id,
//                   s_info->stripe_count
//            );
//        }
//    }
    for (int slave = 1; slave < cfg->world_size; ++slave) {
        get_signal_from_slave(slave);
        char* slave_result_filename = generate_output_filename(output_filename, slave);
        animated_gif* slave_result = load_pixels(slave_result_filename);
        for (int image_index = 0; image_index < cfg->n_images; ++image_index) {
            copy_stripe(
                    slave_result->p[image_index],
                    image->p[image_index],
                    image->width[image_index],
                    image->height[image_index],
                    &cfg->s_info[image_index][slave]);
            free(slave_result->p[image_index]); // We aren't interested on other memory leaks
        }
        remove(slave_result_filename);
        free(slave_result_filename);
    }
}

int master_main(int argc, char *argv[]) {
    char *input_filename;
    char *output_filename;
    animated_gif *image;
    struct timeval t1, t2;
    double duration;

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    /* Check command-line arguments */
    if (argc < 3) {
        fprintf(stderr, "Usage: %s input.gif output.gif\n", argv[0]);
        master_sync(WORK_MODE_FAILURE);
        return 1;
    }

    input_filename = argv[1];
    output_filename = argv[2];

    /* IMPORT Timer start */
    gettimeofday(&t1, NULL);

    /* Load file and store the pixels in array */
    image = load_pixels(input_filename);

    if (image == NULL) {
        return 1;
    }

    /* IMPORT Timer stop */
    gettimeofday(&t2, NULL);

    duration = (t2.tv_sec - t1.tv_sec) + ((t2.tv_usec - t1.tv_usec) / 1e6);

    printf("GIF loaded from file %s with %d image(s) in %lf s\n",
           input_filename, image->n_images, duration);

    /* FILTER Timer start */
    gettimeofday(&t1, NULL);

    bool use_legacy = (world_size >= 4) && (image->n_images >= world_size - 1);

    collection_config* cfg;

    if (use_legacy) {
        master_sync(WORK_MODE_LEGACY);

        cfg = do_master_work_legacy(image);
    } else {
        master_sync(WORK_MODE_STRIPING);

        cfg = do_master_work_striping(image);
    }

    /* FILTER Timer stop */
    gettimeofday(&t2, NULL);

    duration = (t2.tv_sec - t1.tv_sec) + ((t2.tv_usec - t1.tv_usec) / 1e6);

    printf("SOBEL done in %lf s\n", duration);

    /* EXPORT Timer start */
    gettimeofday(&t1, NULL);

    collect_data(image, cfg, output_filename);
    free(cfg);

    /* Store file from array of pixels to GIF file */
    if (!store_pixels(output_filename, image)) {
        return 1;
    }

    /* EXPORT Timer stop */
    gettimeofday(&t2, NULL);

    duration = (t2.tv_sec - t1.tv_sec) + ((t2.tv_usec - t1.tv_usec) / 1e6);

    printf("Export done in %lf s in file %s\n", duration, output_filename);

    return 0;
}

int old_main(int argc, char *argv[]) {
    char *input_filename;
    char *output_filename;
    animated_gif *image;
    struct timeval t1, t2;
    double duration;

    /* Check command-line arguments */
    if (argc < 3) {
        fprintf(stderr, "Usage: %s input.gif output.gif\n", argv[0]);
        return 1;
    }

    input_filename = argv[1];
    output_filename = argv[2];

    /* IMPORT Timer start */
    gettimeofday(&t1, NULL);

    /* Load file and store the pixels in array */
    image = load_pixels(input_filename);

    if (image == NULL) {
        return 1;
    }

    /* IMPORT Timer stop */
    gettimeofday(&t2, NULL);

    duration = (t2.tv_sec - t1.tv_sec) + ((t2.tv_usec - t1.tv_usec) / 1e6);

    printf("GIF loaded from file %s with %d image(s) in %lf s\n",
           input_filename, image->n_images, duration);

    /* FILTER Timer start */
    gettimeofday(&t1, NULL);

    printf("Working mode: single\n");

    for (int i = 0; i < image->n_images; ++i) {
        striping_info s_info;
        s_info.single_mode = 1;
        s_info.min_row = 0;
        s_info.max_row = image->height[i];
        s_info.top_neighbour_id = -1;
        s_info.bottom_neighbour_id = -1;
        s_info.stripe_count = 1;

        apply_all_filters(image, i, &s_info);
    }

    /* FILTER Timer stop */
    gettimeofday(&t2, NULL);

    duration = (t2.tv_sec - t1.tv_sec) + ((t2.tv_usec - t1.tv_usec) / 1e6);

    printf("SOBEL done in %lf s\n", duration);

    /* EXPORT Timer start */
    gettimeofday(&t1, NULL);

    /* Store file from array of pixels to GIF file */
    if (!store_pixels(output_filename, image)) {
        return 1;
    }

    /* EXPORT Timer stop */
    gettimeofday(&t2, NULL);

    duration = (t2.tv_sec - t1.tv_sec) + ((t2.tv_usec - t1.tv_usec) / 1e6);

    printf("Export done in %lf s in file %s\n", duration, output_filename);

    return 0;
}

void report_hostname() {
    char hostname[HOST_NAME_MAX + 1];
    gethostname(hostname, HOST_NAME_MAX + 1);
    printf("hostname: %s\n", hostname);
}

/*
 * Main entry point
 */
int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    prepare_datatypes();

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

#ifdef USE_CUDA
    allocate_device_MPI_process(rank);
    createCudaStreams();
#endif

    // report_hostname();

    if (rank == 0) {
#pragma omp parallel
        {
            if (omp_get_thread_num() == 0)
                printf("Number of MPI processes %d and number of threads %d\n", world_size, omp_get_num_threads());
        }
    }

    int ret_code;

    if (world_size == 1) {
        ret_code = old_main(argc, argv);
    } else if (rank == 0) {
        ret_code = master_main(argc, argv);
    } else {
        ret_code = slave_main(argc, argv);
    }

#ifdef USE_CUDA
    destroyCudaStreams();
#endif

    MPI_Finalize();
    return ret_code;
}
