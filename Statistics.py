#!/usr/bin/env python3
# coding: utf-8


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path


def basename(fname):
    return '.'.join(fname.split('/')[-1].split('.')[:-1])

def main():
    if len(sys.argv) != 2:
        print(f'Usage: {sys.argv[0]} CSV_FILE')
        sys.exit(1)

    csv_filename = sys.argv[1]
    csv_basename = basename(csv_filename)

    df = pd.read_csv(csv_filename)

    image_names = set(df['image_name'])

    for image_name in image_names:
        print("Processing", image_name)
        img_basename = basename(image_name)
        subframe = df[df['image_name'] == image_name]
        image_stats = subframe[['mpi_nprocess', 'omp_nthread', 'nimages', 'sobel_time']].to_numpy()
        n_images = int(image_stats[0][2])

        mpi_nprocess_values = set(image_stats[:,0])


        plt.figure(figsize=(12,6))
        plt.title(f'{image_name}, n_images={n_images}')
        plt.xlabel('Number of threads (OpenMP)')
        plt.ylabel('Time, in seconds')
        for mpi_nprocess in mpi_nprocess_values:
            work_mode = subframe[subframe['mpi_nprocess'] == mpi_nprocess]['work_mode'].values[0]
            data = image_stats[image_stats[:,0] == mpi_nprocess][:,[1,3]]
            data = data[np.argsort(data[:,0])]
            plt.plot(data[:,0], data[:,1], label=f'{int(mpi_nprocess)} processes (MPI, {work_mode})')
        plt.legend()
        plt.ylim(bottom=0)

        dirname = f'stats/{csv_basename}'
        Path(dirname).mkdir(parents=True, exist_ok=True)
        plt.savefig(f'{dirname}/{img_basename}.png')
        # plt.show()

if __name__ == '__main__':
    main()
