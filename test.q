#!/bin/bash --login
#BSUB -J openmpi          # specify job name
#BSUB -o out.o%J          # sets output file
#BSUB -n 58              # specify number of CPU cores required, max 432
#BSUB -R "span[ptile=12]" # specify number of processes per node
#BSUB -x                  # exclusively access to computer nodes

# Usage: bsub < job.q
# Check queue status of your jobs: bjobs
# Check queue status of jobs for all users: bjobs -u all

# load modules
module purge
#module load compiler
#module load interconnect
#module load mpi/openmpi-ft/1.7ft_b3
module list

# some settings
#EXEC_OBJ=mpi_async_1dadvection_compute
GRID_X=12
GRID_Y=12
VERB=0

# OpenMPI requires a hostfile
HOST_FILE=hostfile
echo $LSB_MCPU_HOSTS | awk '{for(i=1;i<NF;i=i+2) {printf "%s slots=%d\n",$i,$(i+1);}}' > $HOST_FILE

# Delete hostfile when shell exits
trap "rm -f $HOST_FILE" EXIT

## If benchmarking then uncomment the following lines to clean the memory on the compute nodes
PDSH_HOSTS=`echo $LSB_MCPU_HOSTS | awk '{for(i=1;i<NF;i=i+2) printf $i",";}'`
pdsh -u 600 -w $PDSH_HOSTS memhog 23g > /dev/null 2>&1

# compile MPI code
#mpicc -O3 -o $EXEC_OBJ ${EXEC_OBJ}.c
make clean
make

# execute MPI code
#mpirun -np 1 --hostfile $HOST_FILE -x LD_LIBRARY_PATH ./$EXEC_OBJ
#mpirun -np 12 --hostfile $HOST_FILE -x LD_LIBRARY_PATH ./$EXEC_OBJ
#mpirun -np 24 --hostfile $HOST_FILE -x LD_LIBRARY_PATH ./$EXEC_OBJ
#mpirun -np 48 --hostfile $HOST_FILE -x LD_LIBRARY_PATH ./$EXEC_OBJ
#mpirun -np 96 --hostfile $HOST_FILE -x LD_LIBRARY_PATH ./$EXEC_OBJ
#mpirun -np 192 --hostfile $HOST_FILE -x LD_LIBRARY_PATH ./$EXEC_OBJ


./run2dAdvect -f -v $VERB -i -p 4  -q 2 -l 6 $GRID_X $GRID_Y ##BSUB -n 58
./run2dAdvect -f -v $VERB -i -p 4  -q 2 -l 6 $GRID_X $GRID_Y ##BSUB -n 58
./run2dAdvect -f -v $VERB -i -p 4  -q 2 -l 6 $GRID_X $GRID_Y ##BSUB -n 58
./run2dAdvect -f -v $VERB -i -p 4  -q 2 -l 6 $GRID_X $GRID_Y ##BSUB -n 58


./run2dAdvect -f -v $VERB -p 4  -q 2 -l 6 $GRID_X $GRID_Y ##BSUB -n 58
