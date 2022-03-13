#!/bin/bash

for i in $(seq 8); do
    for j in $(seq 8); do
        OMP=$i MPI=$j ./run_test_universal.sh
    done
done
