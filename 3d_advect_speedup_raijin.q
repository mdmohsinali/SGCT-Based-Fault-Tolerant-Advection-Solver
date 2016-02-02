#!/bin/bash
#PBS -l ncpus=64
#PBS -q normal
#PBS -l mem=128GB
#PBS -l walltime=00:06:00
#PBS -N SGField
#PBS -M md.ali@anu.edu.au

# best ncpus 64, mem 64GB, walltime 00:01:00 for 9 9 of 2d level 4
# best ncpus 64, mem 128GB, walltime 00:09:00 for 10 10 of 2d level 4
# best ncpus64, mem 128GB, walltime 00:08:00 for 11 11 of 2d level 5
# best ncpus 784, mem 1568GB, walltime 00:02:00 for 10 10 of 2d level 4
# http://nf.nci.org.au/training/NF_IntroCourse.new/slides/allslides.html
# http://nf.nci.org.au/training/NF_IntroCourse.new/
# http://torusware.com/extfiles/doc/fastmpj-documentation/userguide/UsersGuide.pdf

##PBS -P fh0               # specify project fh0
##PBS -l ncpus=32          # specify number of CPU cores required (multiple of 16)
##PBS -q express           # high priority for testing, debugging etc by express
                           # creating benchmark by normal, and another option is copyq
##PBS -l walltime=02:00:00 # specify amount of walltime required hh:mm:ss
##PBS -l mem=220GB         # specify amount of memory required for all nodes (memory for each node x total nodes)
##PBS -N OpenMPI           # specify job name
##PBS -M md.ali@anu.edu.au # specify email address where sending the email after aborting the job

# qsub jobname                # submit jobname job to the queue           
# qstat                       # show the status of the PBS queues
# nqstat                      # enhanced display of the status of the PBS queues
# qstat -s                    # display additional comment on the status of the job
# qps jobid                   # show the processes of a running job
# qls jobid                   # list the files in a job's jobfs directory
# qcat jobid                  # show a running job's stdout, stderr or script
# qcp jobid                   # copy a file from a running job's jobfs directory
# qdel jobid                  # kill a running job
# qdel $(qselect -u username) # deletes all jobs belonging to user username. 
# pbs_rusage jobid            # show the job's current resource usage

# Load modules
#module purge
#module load openmpi
#module list

# Execution info
echo " "
echo "***********************************************************"
echo "BEFORE EXECUTION:"
echo "1. Set LBM parameters in \"input_data\" file. Ignore the"
echo "   values set for \"-da_processors_*\", \"-NZ (for 3D)\"," 
echo "   \"-NX\", \"-NY\". They will be adjusted automatically" 
echo "   on-the-fly and copied to the corresponding directory."
echo "2. Defining \"RUN_ON_COMPUTE_NODES\" in \"FailureRecovery.cpp\""
echo "   file causes running the application on compute nodes"
echo "3. Make sure that settings in \"run3dAdvectRaijin\""
echo "   file is correct."
echo "4. Make sure that \"val_1\" of -p <val_1>, \"val_2\" of" 
echo "   -q <val_2>, and \"val_3\" of -p <val_3> are consistent with"
echo "   \"val_4\" of #BSUB -n <val_4>."
echo "5. Rename \"param_grid_*\" directories that are previously"
echo "   generated and needed in future. Otherwise, they will be"
echo "   deleted automatically during application execution."
echo "6. Run \"./app_raijin_cluster -h\" to get help how to execute"
echo "   (when make completes)."
echo "***********************************************************"
echo " "

# Some job information
echo PBS: qsub is running on $PBS_O_HOST
echo PBS: originating queue is $PBS_O_QUEUE
echo PBS: executing queue is $PBS_QUEUE
echo PBS: working directory is $PBS_O_WORKDIR
echo PBS: execution mode is $PBS_ENVIRONMENT
echo PBS: job identifier is $PBS_JOBID
echo PBS: job name is $PBS_JOBNAME
echo PBS: current home directory is $PBS_O_HOME
echo PBS: PATH = $PBS_O_PATH

# Chnage the directory to current working directory 
cd $PBS_O_WORKDIR

# Some settings
# For 2D: -NX 2^X_DIM, -NY 2^Y_DIM
# For 3D: -NX 2^X_DIM, -NY 2^Y_DIM, -NZ 2^Z_DIM
X_DIM=8
Y_DIM=8
Z_DIM=0
LEVEL=4

# Open MPI requires a hostfile
HOST_FILE=hostfile

for i in $(sort -u $PBS_NODEFILE)
   do
   slots=0
   for j in $(cat $PBS_NODEFILE)
      do
      if [ "$i" == "$j" ];
      then
         slots=`expr $slots + 1`
      fi
      done
   echo "$i slots=$slots" >> $HOST_FILE
done

# Delete hostfile when shell exits
trap "rm -f $HOST_FILE" EXIT

# Total number of processes for nodes on hostfile
#NPROCS=$PBS_NCPUS

# Compile Fortran and MPI codes
(pwd;\
#echo "Setting environment from \"set_environment.raijin_cluster.non_ft_gfortran.sh\"";\
#source set_environment.raijin_cluster.non_ft_gfortran.sh;\
#echo "Calling make allclean; make";\
#make allclean;\
make;\
pwd)

# Manually map MPI ranks to CPU cores (calculation is based on hostfile)
#chmod 755 mapRanks.sh
#./mapRanks.sh 2 # either 2D (pass parameter 2) or 3D (pass parameter 3)

# Execute MPI code
# For fixed-time steps
#./run3dAdvectRaijin -v 1 -f -p 128 -q 64 -r 32 -l ${LEVEL} ${X_DIM} ${Y_DIM} ${Z_DIM} # for 2D for 784 level 4
./run3dAdvectRaijin -v 0 -f -p 8 -q 4 -r 2 -l ${LEVEL} ${X_DIM} ${Y_DIM} ${Z_DIM} # for 2D for 49 level 4, 64 level 5

#qstat -f $PBS_JOBID | grep used

# Unset flags
module purge
