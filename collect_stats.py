#!/usr/bin/env python3

# Sample input:
# Running test on images/original/051009.vince.gif -> images/processed/051009.vince-sobel.gif
# Number of MPI processes 1 and number of threads 1
# GIF loaded from file images/original/051009.vince.gif with 1 image(s) in 0.018448 s
# Working mode: striping
# SOBEL done in 0.047131 s
# Export done in 0.259479 s in file images/processed/051009.vince-sobel.gif

import csv
import sys
import itertools as it

BLOCK_SIZE = 6

class Statistics:
    def __init__(self):
        self.head: Tuple[str, str, str, str, str, str, str, str] = [
                'image_name',
                'mpi_nprocess',
                'omp_nthread',
                'nimages',
                'load_time',
                'sobel_time',
                'store_time',
                'work_mode'
            ]
        self.data: List[Tuple[str, str, str, str, str, str, str, str]] = []

    def read_file(self, filename):
        with open(filename, 'r') as f:
            while True:
                block = list(it.islice(f, BLOCK_SIZE))
                if len(block) < BLOCK_SIZE:
                    break
                self.process_block(block)

    def process_block(self, block):
        print(block)
        # Hardcoded stuff, sorry
        image_name      = block[0].split()[3]
        mpi_nprocess    = block[1].split()[4]
        omp_nthread     = block[1].split()[-1]
        nimages         = block[2].split()[-5]
        load_time       = block[2].split()[-2]
        sobel_time      = block[4].split()[-2]
        store_time      = block[5].split()[3]
        work_mode       = block[3].split()[-1]

        self.data.append((image_name, mpi_nprocess, omp_nthread, nimages, load_time, sobel_time, store_time, work_mode))

    def export_statistics(self, filename):
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(self.head)
            writer.writerows(self.data)


def main():
    statistics = Statistics()
    for filename in sys.argv[1:]:
        statistics.read_file(filename)
    statistics.export_statistics('statistics.csv')

if __name__ == '__main__':
    main()
