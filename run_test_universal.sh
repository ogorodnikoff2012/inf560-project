#!/bin/bash

if [ -z "$OMP" -o -z "$MPI" ]; then
    echo "You have to set environment variables OMP and MPI"
    exit 1
fi

# make -j8

INPUT_DIR=images/original
OUTPUT_DIR=images/processed
LOG_DIR=logs

mkdir $OUTPUT_DIR 2>/dev/null
mkdir $LOG_DIR 2>/dev/null

SCRIPT_FILE="$(basename $0)"
LOG_FILE="${LOG_DIR}/run_test_omp_${OMP}_mpi_${MPI}.log"

rm -r "$LOG_FILE" 2>/dev/null

for i in $INPUT_DIR/*gif ; do
    DEST=$OUTPUT_DIR/`basename $i .gif`-sobel.gif
    echo "Running test on $i -> $DEST" | tee -a "$LOG_FILE"

    # salloc -N 16 -n 16 srun sh -c 'echo $(hostname) $SLURM_CPUS_ON_NODE'

    # OMP_NUM_THREADS="${OMP}" GOMP_CPU_AFFINITY=0-7 salloc -c "$OMP" -n "${MPI}" -N "$MPI" mpirun ./sobelf $i $DEST | tee -a "$LOG_FILE"
    # OMP_NUM_THREADS="${OMP}" salloc --threads-per-core=1 -c "$OMP" -n "${MPI}" -N "$MPI" scalasca -analyze mpirun ./sobelf $i $DEST | tee -a "$LOG_FILE"
    OMP_NUM_THREADS="${OMP}" salloc --threads-per-core=1 -c "$OMP" -n "${MPI}" -N "$MPI" mpirun ./sobelf $i $DEST | tee -a "$LOG_FILE"
done
