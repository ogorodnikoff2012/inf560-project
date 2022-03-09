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

/* Set this macro to 1 to enable debugging information */
#define SOBELF_DEBUG 0

/* Represent one pixel from the image */
typedef struct pixel {
    uint8_t r; /* Red */
    uint8_t g; /* Green */
    uint8_t b; /* Blue */
} pixel;

MPI_Datatype kMPIPixelDatatype;

/* Represent one GIF image (animated or not */
typedef struct animated_gif {
    int n_images; /* Number of images */
    int *width; /* Width of each image */
    int *height; /* Height of each image */
    pixel **p; /* Pixels of each image */
    GifFileType *g; /* Internal representation.
                         DO NOT MODIFY */
} animated_gif;

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

void apply_gray_filter(animated_gif *image, int image_index) {
    int j;
    pixel *p;

    p = (image->p)[image_index];

    #pragma omp parallel for schedule(static)
    for (j = 0; j < image->width[image_index] * image->height[image_index]; j++) {
        int moy;

        moy = (p[j].r + p[j].g + p[j].b) / 3;

        if (moy < 0) {
            moy = 0;
        }

        if (moy > 255) {
            moy = 255;
        }

        p[j].r = moy;
        p[j].g = moy;
        p[j].b = moy;
    }
}

#define CONV(l, c, nb_c) \
    ((l)*(nb_c)+(c))

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

void apply_blur_filter(animated_gif *image, int size, int threshold, int image_index) {
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
            #pragma omp parallel
            {
                #pragma omp for collapse(2) schedule(static)
                for (j = 0; j < height - 1; j++) {
                    for (k = 0; k < width - 1; k++) {
                        new[CONV(j, k, width)].r = p[CONV(j, k, width)].r;
                        new[CONV(j, k, width)].g = p[CONV(j, k, width)].g;
                        new[CONV(j, k, width)].b = p[CONV(j, k, width)].b;
                    }
                }

                /* Apply blur on top part of image (10%) */
                #pragma omp for collapse(2) schedule(static)
                for (j = size; j < height / 10 - size; j++) {
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
                const int finish = height * 0.9 + size;
                #pragma omp for collapse(2) schedule(static)
                for (j = height / 10 - size; j < finish; j++) {
                    for (k = size; k < width - size; k++) {
                        new[CONV(j, k, width)].r = p[CONV(j, k, width)].r;
                        new[CONV(j, k, width)].g = p[CONV(j, k, width)].g;
                        new[CONV(j, k, width)].b = p[CONV(j, k, width)].b;
                    }
                }

                /* Apply blur on the bottom part of the image (10%) */
                #pragma omp for collapse(2) schedule(static)
                for (j = height * 0.9 + size; j < height - size; j++) {
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

                #pragma omp for collapse(2) schedule(static)
                for (j = 1; j < height - 1; j++) {
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
            }
        } while (threshold > 0 && !end);
    }
#if SOBELF_DEBUG
    printf( "BLUR: number of iterations for image %d\n", n_iter ) ;
#endif

    free(new);

}

void apply_sobel_filter(animated_gif *image, int image_index) {
    int j, k;
    int width, height;

    pixel *p;

    p = (image->p)[image_index];

    width = image->width[image_index];
    height = image->height[image_index];

    pixel *sobel;

    sobel = (pixel *) malloc(width * height * sizeof(pixel));

    #pragma omp parallel
    {
        #pragma omp for collapse(2) schedule(static)
        for (j = 1; j < height - 1; j++) {
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

        #pragma omp for collapse(2) schedule(static)
        for (j = 1; j < height - 1; j++) {
            for (k = 1; k < width - 1; k++) {
                p[CONV(j, k, width)].r = sobel[CONV(j, k, width)].r;
                p[CONV(j, k, width)].g = sobel[CONV(j, k, width)].g;
                p[CONV(j, k, width)].b = sobel[CONV(j, k, width)].b;
            }
        }
    }
    free(sobel);


}

void apply_all_filters(animated_gif *image) {
    for (int i = 0; i < image->n_images; ++i) {
        // Convert the pixels into grayscale
        apply_gray_filter(image, i);
    }

    for (int i = 0; i < image->n_images; ++i) {
        // Apply blur filter with convergence value
        apply_blur_filter(image, 5, 20, i);
    }

    for (int i = 0; i < image->n_images; ++i) {
        // Apply sobel filter on pixels
        apply_sobel_filter(image, i);
    }
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

int slave_sync(void) {
    int value;
    MPI_Bcast(&value, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return value;
}

void master_sync(int value) {
    MPI_Bcast(&value, 1, MPI_INT, 0, MPI_COMM_WORLD);
}

int slave_main(int argc, char *argv[]) {
    int rank;
    int world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    animated_gif image;

    if (!slave_sync()) {
        return 1;
    }

    /* First, we broadcast metadata */
    MPI_Bcast(&image.n_images, 1, MPI_INT, 0, MPI_COMM_WORLD);

    image.width = calloc(image.n_images, sizeof(int));
    image.height = calloc(image.n_images, sizeof(int));
    image.p = calloc(image.n_images, sizeof(pixel *));

    MPI_Bcast(image.width, image.n_images, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(image.height, image.n_images, MPI_INT, 0, MPI_COMM_WORLD);

    const int kSignalTag = image.n_images;

    int image_index = -1;
    MPI_Send(&image_index, 1, MPI_INT, 0, kSignalTag, MPI_COMM_WORLD);

    MPI_Request processed_image_requests[image.n_images];
    for (int i = 0; i < image.n_images; ++i) {
        processed_image_requests[i] = MPI_REQUEST_NULL;
    }

    while (true) {
        MPI_Recv(&image_index, 1, MPI_INT, 0, kSignalTag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (image_index == -1) {
            break;
        }

        image.p[image_index] = calloc(image.width[image_index] * image.height[image_index], sizeof(pixel));
        MPI_Recv(image.p[image_index], image.width[image_index] * image.height[image_index], kMPIPixelDatatype, 0,
                 image_index, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        /* Convert the pixels into grayscale */
        apply_gray_filter(&image, image_index);

        /* Apply blur filter with convergence value */
        apply_blur_filter(&image, 5, 20, image_index);

        /* Apply sobel filter on pixels */
        apply_sobel_filter(&image, image_index);


        MPI_Request req;
        MPI_Isend(&image_index, 1, MPI_INT, 0, kSignalTag, MPI_COMM_WORLD, &req);
        MPI_Request_free(&req);

        MPI_Isend(image.p[image_index], image.width[image_index] * image.height[image_index], kMPIPixelDatatype, 0,
                  image_index, MPI_COMM_WORLD, processed_image_requests + image_index);
    }

    for (int i = 0; i < image.n_images; ++i) {
        if (processed_image_requests[i] != MPI_REQUEST_NULL) {
            MPI_Wait(processed_image_requests + i, MPI_STATUS_IGNORE);
        }
        free(image.p[i]);
    }

    free(image.width);
    free(image.height);
    free(image.p);

    return 0;
}

void do_master_work(animated_gif *image) {
    int rank;
    int world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    const int kSignalTag = image->n_images;

    /* First, we broadcast metadata */
    MPI_Bcast(&image->n_images, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(image->width, image->n_images, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(image->height, image->n_images, MPI_INT, 0, MPI_COMM_WORLD);

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
            MPI_Irecv(image->p[image_index], image->width[image_index] * image->height[image_index],
                      kMPIPixelDatatype, indx, image_index, MPI_COMM_WORLD, &processed_image_requests[image_index]);
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
            MPI_Isend(image->p[next_image_index], image->width[next_image_index] * image->height[next_image_index],
                      kMPIPixelDatatype, indx, next_image_index, MPI_COMM_WORLD, &req);
            MPI_Request_free(&req);

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
}

int master_main(int argc, char *argv[]) {
    char *input_filename;
    char *output_filename;
    animated_gif *image;
    struct timeval t1, t2;
    double duration;

    /* Check command-line arguments */
    if (argc < 3) {
        fprintf(stderr, "Usage: %s input.gif output.gif\n", argv[0]);
        master_sync(0);
        return 1;
    }

    master_sync(1);

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

    do_master_work(image);

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

    apply_all_filters(image);

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

    prepare_pixel_datatype(&kMPIPixelDatatype);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

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

    MPI_Finalize();
    return ret_code;
}
