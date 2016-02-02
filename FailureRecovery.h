/* SGCT-Based-Fault-Tolerant-Advection-Solver Code source file.
   Copyright (c) 2015, Md Mohsin Ali. All rights reserved.
   Licensed under the terms of the BSD License as described in the LICENSE_FT_CODE file.
   This comment must be retained in any redistributions of this source file.
*/

/* 
 * File       : FailureRecovery.h
 * Description: contains functions for reconstructing faulty communicator by
 *              performing in-order failed process replacement, and other 
 *              supporting functions to achieve this
 * Author     : Mohsin Ali
 * Created    : August 2013
 * Updated    : May 2014, June 2014
 */


#ifndef FAILURERECOVERY_INCLUDED
#define FAILURERECOVERY_INCLUDED

// Header files
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
#include "mpi.h"
#include "mpi-ext.h"
#ifdef _OPENMP
#include <omp.h>
#endif


// Defined values
#define HANG_ON_REMOVE           // defining this remove fault-tolerant mpi hang on
#define SLOTS                 getSlots()
#define RECOV_ON_SPARE_NODES     // defining this causes processes will be spawned on spare nodes
                                 // (handling node failures),
                                 // otherwise, spawned on the same node where it was before
                                 // the failure (handling process failures)
#define RUN_ON_COMPUTE_NODES     // defining this causes run on compute nodes, otherwise,
                                 // test run on head node/personal machine
#define GLOBAL_DETECTION         // defining this causes global detection, otherwise, local detection

#define MERGE_TAG             20000
#define MERGE_TAG2            20001
#define NON_FAILED_FAILED_TAG 20002
#define PROCS_GRID_NF_TAG     20003
#define PROCS_GRID_NF_TAG2    20004

// Function prototype
MPI_Comm communicatorReconstruct(MPI_Comm myCommWorld, int childFlag, int * listFails, int * numFails,
                int * numNodeFails, int sumPrevNumNodeFails, int argc, char ** argv, int verbosity);
int numProcsFails(MPI_Comm comm);
void mpiErrorHandler(MPI_Comm * comm, int *errorCode, ...);
void repairComm(MPI_Comm * broken, MPI_Comm * repaired, int iteration, int * listFails, int * numFails,
                int * numNodeFails, int sumPrevNumNodeFails, int argc, char ** argv, int verbosity);
int rankIsNotOnFailedList(int rank, int * failedList, int numFails);
int getHostfileLastLineIndex(void);
int getSlots(void);
char * getHostToLaunch(int hostfileLineIndex);
int ranksOnSameGrid(int rank, int numFails, int * listFails, GridCombine2D * gc);
int isPreviouslySend(int currentIndex, int * listFails, GridCombine2D * gc);
int minRank(int fGid, int nprocs, GridCombine2D * gc);
int isRightBoundary(Vec2D<int> id, Vec2D<int> P);
int isUpperBoundary(Vec2D<int> id, Vec2D<int> P);
int getToRank(int min_rank, int fidx, int fidy, int fPx);
double twoPow(int x);
double ** alloc2dDouble(int rows, int cols);
void recoveryByResampling(HaloArray2D *u, ProcGrid2D *g, MPI_Comm myCommWorld, int isParent, 
                 GridCombine2D *gc, int rank, int nprocs, int level, int numFails, int * listFails);
void duplicateToSpareGrids(HaloArray2D *u, ProcGrid2D *g, MPI_Comm myCommWorld, GridCombine2D *gc, 
                 int rank, int nprocs, int level);
void checkpointWrite2D(HaloArray2D *u, int rank);
void checkpointRead2D(HaloArray2D *u, int rank);

/////////////////////////////////////////////////////////////////////////////////////////////////////
MPI_Comm communicatorReconstruct(MPI_Comm myCommWorld, int childFlag, int * listFails, int * numFails,
        int * numNodeFails, int sumPrevNumNodeFails, int argc, char ** argv, int verbosity){
	int i, ret, rank, nprocs, oldRank = 0, totFails = 0, * failedList, flag;
        int iterCounter = 0, failure = 0, recvVal[2], length;
        MPI_Status mpiStatus;
	MPI_Comm parent, mcw;
	MPI_Comm dupComm, tempIntracomm, unorderIntracomm;
        MPI_Errhandler newEh;
        double startTime = 0.0, endTime;
        char hostName[MPI_MAX_PROCESSOR_NAME];

        // Error handler
        MPI_Comm_create_errhandler(mpiErrorHandler, &newEh);

	MPI_Comm_get_parent(&parent);

        MPI_Comm_rank(myCommWorld, &rank);
        if(MPI_COMM_NULL == parent && childFlag == 0 && rank == 0){
           startTime = MPI_Wtime();
        }

        do{
           failure = 0;
           ret = MPI_SUCCESS;
           /*
           if(childFlag == 0 && MPI_COMM_NULL != parent){
              parent = MPI_COMM_NULL;
           }
           */
           // Parent part
	   if(MPI_COMM_NULL == parent){
              if(iterCounter == 0){
                 mcw = myCommWorld;
              }
              // Set error handler for communicator
              MPI_Comm_set_errhandler(mcw, newEh);

              // World information
              MPI_Comm_rank(mcw, &rank);
	      MPI_Comm_size(mcw, &nprocs);
              // Synchronize. Sometimes hangs on without this
#ifdef HANG_ON_REMOVE
              //MPI_Barrier(mcw);
              OMPI_Comm_agree(mcw, &flag); // since some of the times MPI_Barrier hangs on
#endif             

              // Target function
	      //if(MPI_SUCCESS != (ret = MPI_Barrier(mcw))){
              if(MPI_SUCCESS != (ret = MPI_Comm_dup(mcw, &dupComm))){
                 if(verbosity > 0 && rank == 0){
                    printf("[????? Process %d (nprocs %d)] MPI_Comm_dup (parent): "
                           "Unsuccessful (due to process failure) OK\n", rank, nprocs);
                 }

                 // Revoke the communicator
	         if(MPI_SUCCESS != (OMPI_Comm_revoke(mcw))){
                    if(rank == 0){
                       printf("[Process %d (nprocs %d)] Iteration %d: OMPI_Comm_revoke "
                              "(parent): Error!\n", rank, nprocs,  iterCounter);
                    }
                 }
                 else{
                    if(verbosity > 1 && rank == 0){
                       printf("[Process %d (nprocs %d)] Iteration %d: OMPI_Comm_revoke "
                              "(parent): SUCCESS\n", rank, nprocs, iterCounter);
                    }
                 }

                 // Call repair with splitted world
                 totFails = numProcsFails(mcw);
                 failedList = (int *) malloc(totFails*sizeof(int));
                 repairComm(&mcw, &tempIntracomm, iterCounter, failedList, numFails, numNodeFails,
                            sumPrevNumNodeFails, argc, argv, verbosity);

                 // Assign list of failed processes
                 #pragma omp parallel for default(shared)
                 for(i = 0; i < *numFails; i++){
                    listFails[i] = failedList[i];
                 }
                 // Free memory
                 free(failedList);

                 // Operation failed: retry
                 failure = 1;
              } //end of "if MPI_Barrier/MPI_Comm_dup fails"
              else{
                 if(verbosity > 0 && rank == 0){
                    printf("[..... Process %d (nprocs %d)] Iteration %d: MPI_Comm_dup "
                           "(parent): SUCCESS\n", rank, nprocs, iterCounter);
                 }

                 // Operation success: breaking iteration
                 failure = 0;
              }
	   } // end of "parent"
           // Child part
	   else{
              MPI_Comm_set_errhandler(parent, newEh);
              // Synchronize. Sometimes hangs on without this
              // Position of code and intercommunicator, parent, (not intra) is important
#ifdef HANG_ON_REMOVE
              //MPI_Barrier(parent);
              OMPI_Comm_agree(parent, &flag);// since some of the times MPI_Barrier hangs on
#endif              

              MPI_Comm_rank(parent, &rank);
              MPI_Comm_size(parent, &nprocs);

              if(verbosity > 0 && rank == 0){
                  MPI_Get_processor_name(hostName, &length);
                  printf("[Process %d, nprocs = %d] created on host %s (child)\n", 
                          rank, nprocs, hostName);
              }

              if(MPI_SUCCESS != (MPI_Intercomm_merge(parent, true, &unorderIntracomm))){
                 if(rank == 0){
                    printf("[Process %d] Iteration %d: MPI_Intercomm_merge (child): Error!\n", 
                           rank, iterCounter);
                 }
              }
              else{
                 if(verbosity > 1 && rank == 0){
                    printf("[Process %d] Iteration %d: MPI_Intercomm_merge (child): SUCCESS\n", 
                           rank, iterCounter);
                 }
              }
              // Receive failed ranks and number of fails from process 0 of parent
              if(MPI_SUCCESS != (MPI_Recv(&recvVal, 2, MPI_INT, 0, MERGE_TAG, 
                                unorderIntracomm, &mpiStatus))){
                 if(rank == 0){
                    printf("[Process %d] Iteration %d: MPI_Recv1 (child): Error!\n", 
                           rank, iterCounter);
                 }
              }
              else{
                 if(verbosity > 1 && rank == 0){
                    printf("[Process %d] Iteration %d: MPI_Recv1 (child): SUCCESS\n", 
                           rank, iterCounter);
                 }
                 oldRank = recvVal[0]; *numFails = recvVal[1];
              }
              
              // Split the communicator to order the ranks.
              // No order is maintaining here. Actual ordering is done on parent side
              // This is a support only to parent side
              if(MPI_SUCCESS != (MPI_Comm_split(unorderIntracomm, 0, oldRank, &tempIntracomm))){
                 if(rank == 0){
                    printf("[Process %d] Iteration %d: MPI_Comm_split (child): Error!\n", 
                           rank, iterCounter);
                 }
              }
              else{
                 if(verbosity > 1 && rank == 0){
                    printf("[Process %d] Iteration %d: MPI_Comm_split (child): SUCCESS\n", 
                           rank, iterCounter);
                 }
              }

              // Operation on parent failed: retry
              ret = (!MPI_SUCCESS);
              failure = 1;

              // Free memory
              MPI_Comm_free(&unorderIntracomm);
              MPI_Comm_free(&parent);
	   }// end of "child"

           // Reset comm world
           if(ret != MPI_SUCCESS){
              mcw = tempIntracomm;
           }

           // Reset parent value for parent
           if(parent == MPI_COMM_NULL && ret != MPI_SUCCESS){
              parent = mcw;
           }

           // Reset parent value of child and make the operation collective
           if(MPI_SUCCESS != ret && MPI_COMM_NULL != parent){
              parent = MPI_COMM_NULL;
           }
           iterCounter++;
        }while(failure > 1);// replace 'failure > 1' with 'failure' if want fault tolerant recovery

        if(MPI_COMM_NULL == parent && childFlag == 0 && rank == 0){
           endTime = MPI_Wtime();
           printf("[%d]----- Reconstructing failed communicator (including failed list creation) "
                  "takes %0.6f Sec (MPI_Wtime) -----\n", rank, endTime - startTime);
        }

        // Memory release
        MPI_Errhandler_free(&newEh);

        return mcw;
}//communicatorReconstruct()

///////////////////////////////////////////////////////////////////////////////////
int numProcsFails(MPI_Comm mcw){
	int rank;
        MPI_Group fGroup;
        int ret, numFailures = 0, flag;
        MPI_Errhandler newEh;

        MPI_Comm dupComm;

        // error handler
        MPI_Comm_create_errhandler(mpiErrorHandler, &newEh);

        MPI_Comm_rank(mcw, &rank);

        // set error handler for communicator
        MPI_Comm_set_errhandler(mcw, newEh);

        // target function
	//if(MPI_SUCCESS != (ret = MPI_Barrier(mcw))){
        if(MPI_SUCCESS != (ret = MPI_Comm_dup(mcw, &dupComm))){

           OMPI_Comm_failure_ack(mcw);
           OMPI_Comm_failure_get_acked(mcw, &fGroup);

           // Get the number of failures
           MPI_Group_size(fGroup, &numFailures);

        }// end of "MPI_Barrier/MPI_Comm_dup failure"

        OMPI_Comm_agree(mcw, &flag);

        // memory release
	if(numFailures > 0){
           MPI_Group_free(&fGroup);
        }
        MPI_Errhandler_free(&newEh);

        return numFailures;
}//numProcsFails()


/////////////////////////////////////////////////////////////////////////////////////////////
void mpiErrorHandler(MPI_Comm * comm, int *errorCode, ...){
    MPI_Group failedGroup;

    OMPI_Comm_failure_ack(*comm);
    OMPI_Comm_failure_get_acked(*comm, &failedGroup);

    /*
    int rank, nprocs;
    MPI_Comm_rank(*comm, &rank);
    MPI_Comm_size(*comm, &nprocs);

    if(*errorCode == MPI_ERR_PROC_FAILED ) {
       printf("(Process %d of %d) Error Handler: MPI_ERROR_PROC_FAILED Detected.\n", rank, nprocs);
    } else {
       printf("(Process %d of %d) Error Handler: Other Failure Detected.\n", rank, nprocs);
    }
    */

    // Introduce a small delay to aid debugging
    fflush(NULL);

    // (1) Without delay, failed processes will NOT be synchronized.
    // (2) This delay MUST be through error handler. Otherwise, NOT work.
    // (3) 10 milliseconds is the minimum delay I have tested on a dual-core machine
    //     for 200 processes. Not working if smaller time is given.

    //sleep(1);    // 1 second
    usleep(10000); // 10 milliseconds
                   // MPI_Comm_revoke is failed without this for a large number of processes, but works for 1 process failure.
                   // most of the time hangs on for more than 1 process failure without this.
                   // necessary to enable this for a small number of processes to test.

    MPI_Group_free(&failedGroup);

    return;
}//mpiErrorHandler()


///////////////////////////////////////////////////////////////////////////////////////////////////////
void repairComm(MPI_Comm * broken, MPI_Comm * repaired, int iteration, int * listFails, int * numFails,
        int * numNodeFails, int sumPrevNumNodeFails, int argc, char ** argv, int verbosity){
	MPI_Comm tempShrink, unorderIntracomm, tempIntercomm;
	int i, ret, result, procsNeeded = 0, oldRank, newRank, oldGroupSize, rankKey = 0, flag;
	int * tempRanks, * failedRanks, * errCodes, rank, hostfileLineIndex;
	MPI_Group oldGroup, failedGroup, shrinkGroup;
        int hostfileLastLineIndex, tempLineIndex, * failedNodeList = NULL, * nodeList = NULL, totNodeFailed = 0;
        double startTime = 0.0, endTime;
        int nprocs, j, * shrinkMergeList;
        char hostName[128];
        gethostname(hostName, sizeof(hostName));

        char ** appToLaunch;
        char *** argvToLaunch;
        int * procsNeededToLaunch;
        MPI_Info * hostInfoToLaunch;
        char ** hostNameToLaunch;

        MPI_Comm_rank(*broken, &rank);
        if(rank == 0){
            startTime = MPI_Wtime();
        }

#ifndef GLOBAL_DETECTION
	MPI_Comm_size(*broken, &oldGroupSize);
	MPI_Comm_group(*broken, &oldGroup);
	MPI_Comm_rank(*broken, &oldRank);
	OMPI_Comm_failure_ack(*broken);
	OMPI_Comm_failure_get_acked(*broken, &failedGroup);
	MPI_Group_size(failedGroup, &procsNeeded);
	errCodes = (int *) malloc(sizeof(int) * procsNeeded);

	// Figure out ranks of the processes which had failed
	tempRanks = (int *) malloc(sizeof(int) * oldGroupSize);
	failedRanks = (int *) malloc(sizeof(int) * oldGroupSize);
	#pragma omp parallel for default(shared)
	for(i = 0; i < oldGroupSize; i++) {
	   tempRanks[i] = i;
	}

	MPI_Group_translate_ranks(failedGroup, procsNeeded, tempRanks, oldGroup, failedRanks);
#endif        

        double shrinkTime = MPI_Wtime();
        // Shrink the broken communicator to remove failed procs
	if(MPI_SUCCESS != (ret = OMPI_Comm_shrink(*broken, &tempShrink))){
           printf("Iteration %d: OMPI_Comm_shrink (parent): ERROR!\n", iteration);
        }
        else{
           if(verbosity > 1 ){
              printf("Iteration %d: OMPI_Comm_shrink (parent): SUCCESS\n", iteration);
           }
        }
        if (verbosity > 0 && rank == 0){
           printf("OMPI_Comm_shrink takes %0.6f Sec\n", MPI_Wtime() - shrinkTime);
        }

#ifdef GLOBAL_DETECTION
	MPI_Comm_group(*broken, &oldGroup);
	MPI_Comm_group(tempShrink, &shrinkGroup);
	MPI_Comm_size(*broken, &oldGroupSize);

	MPI_Group_compare(oldGroup, shrinkGroup, &result);

	if(result != MPI_IDENT){
	   MPI_Group_difference(oldGroup, shrinkGroup, &failedGroup);
	}

	MPI_Comm_rank(*broken, &oldRank);
	MPI_Group_size(failedGroup, &procsNeeded);

	errCodes = (int *) malloc(sizeof(int)*procsNeeded);

	// Figure out ranks of the processes which had failed
	tempRanks = (int*)malloc(sizeof(int)*oldGroupSize);
	failedRanks = (int*)malloc(sizeof(int)*oldGroupSize);
	#pragma omp parallel for default(shared)
	for(i = 0; i < oldGroupSize; i++){
	   tempRanks[i] = i;
	}

	MPI_Group_translate_ranks(failedGroup, procsNeeded, tempRanks, oldGroup, failedRanks);

	MPI_Group_free(&shrinkGroup);
#endif        

        // Assign number of failed processes
        *numFails = procsNeeded;

        hostNameToLaunch = (char **) malloc(procsNeeded * sizeof(char *));

        if(verbosity > 0 && rank == 0){
	       printf("*** Iteration %d: Application: Number of process(es) failed in the corresponding "
                      "communicator is %d ***\n", iteration, procsNeeded);
        }

        if(rank == 0){
            endTime = MPI_Wtime();
            printf("[%d]----- Creating failed process list takes %0.6f Sec (MPI_Wtime) -----\n", rank, endTime - startTime);
        }

#ifdef RECOV_ON_SPARE_NODES
	// Determining total number of node failed, and a list of them
        hostfileLastLineIndex = getHostfileLastLineIndex(); //started from 0
	nodeList = (int *) malloc((hostfileLastLineIndex+1) * sizeof(int));
	memset(nodeList, 0, (hostfileLastLineIndex+1)*sizeof(int)); // initialize nodeList with 0's
	      
	for(int i = 0; i < procsNeeded; ++i){
	   tempLineIndex = failedRanks[i]/SLOTS; //started from 0
	   nodeList[tempLineIndex] = 1;
	}

	for(int nodeCounter = 0; nodeCounter < (hostfileLastLineIndex+1); ++nodeCounter){
	   totNodeFailed += nodeList[nodeCounter];
	}
        *numNodeFails = totNodeFailed;

	// Check if there is sufficient spare node available for recovery
        if((hostfileLastLineIndex - totNodeFailed -sumPrevNumNodeFails) < (oldGroupSize-1)/SLOTS){
           if(rank == 0){
              printf("[%d] There is no sufficient spare node available for recovery.\n", rank);
           }
           exit(0);
        }

	failedNodeList = (int *) malloc(totNodeFailed * sizeof(int));
	memset(failedNodeList, 0, totNodeFailed * sizeof(int)); // initialize failedNodeList with 0's

	int failedNodeCounter = 0;
	for(int nodeCounter = 0; nodeCounter < (hostfileLastLineIndex+1); ++nodeCounter){
	   if(nodeList[nodeCounter] == 1){
	      failedNodeList[failedNodeCounter++] = nodeCounter;
	   }
	}
#endif

        char * hostNameFailed = NULL;
        #pragma omp parallel for default(shared)
	for(i = 0; i < procsNeeded; ++i){
           // Assign list of processes failed
           listFails[i] = failedRanks[i];

#ifdef RUN_ON_COMPUTE_NODES
	   tempLineIndex = failedRanks[i]/SLOTS; //started from 0
#ifdef RECOV_ON_SPARE_NODES
           for(int j = 0; j < totNodeFailed; ++j){
              if(failedNodeList[j] == tempLineIndex){
                 hostfileLineIndex = hostfileLastLineIndex - j - sumPrevNumNodeFails;
              }
           }
#else // Recovery on the same node (no node failure, only process failure)
	   hostfileLineIndex = tempLineIndex;	      
#endif
	   hostNameToLaunch[i] = getHostToLaunch(hostfileLineIndex);
           hostNameFailed = getHostToLaunch(tempLineIndex);           
#else // Run on head node or personal machine
           hostNameToLaunch[i] = (char *)hostName;
           hostNameFailed = (char *)hostName;
#endif           

           if(verbosity > 0 && rank == 0){
	      printf("--- Iteration %d: Application: Process %d on node %s is failed! ---\n", iteration, failedRanks[i], hostNameFailed);
	   }
        }
        // Release memory of hostNameFailed
        free(hostNameFailed);

        appToLaunch = (char **) malloc(procsNeeded * sizeof(char *));
        argvToLaunch = (char ***) malloc(procsNeeded * sizeof(char **));
        procsNeededToLaunch = (int *) malloc(procsNeeded * sizeof(int));
        hostInfoToLaunch = (MPI_Info *) malloc(procsNeeded * sizeof(MPI_Info));
        argv[argc] = NULL;
        #pragma omp parallel for default(shared)
        for(i = 0; i < procsNeeded; i++){
            appToLaunch[i] = (char *)argv[0];
            argvToLaunch[i] = (char **)argv;
            procsNeededToLaunch[i] = 1;
            // Host information where to spawn the processes
            MPI_Info_create(&hostInfoToLaunch[i]);
            MPI_Info_set(hostInfoToLaunch[i], (char *)"host", hostNameToLaunch[i]);
            //MPI_Info_set(hostInfoToLaunch[i], "hostfile", "hostfile");
        }

        double spawnTime = MPI_Wtime();
#ifdef HANG_ON_REMOVE
        OMPI_Comm_agree(tempShrink, &flag);
#endif
	// Spawn the new process(es)
	if(MPI_SUCCESS != (ret = MPI_Comm_spawn_multiple(procsNeeded, appToLaunch, argvToLaunch, procsNeededToLaunch,
	   hostInfoToLaunch, 0, tempShrink, &tempIntercomm, MPI_ERRCODES_IGNORE))){
	   free(tempRanks);
	   free(failedRanks);
	   free(errCodes);
	   if(MPI_ERR_COMM  == ret){
	      printf("Iteration %d: MPI_Comm_spawn_multiple: Invalid communicator (parent)\n", iteration);
	   }
	   if(MPI_ERR_ARG  == ret){
	      printf("Iteration %d: MPI_Comm_spawn_multiple: Invalid argument (parent)\n", iteration);
	   }
	   if(MPI_ERR_INFO  == ret){
	       printf("Iteration %d: MPI_Comm_spawn_multiple: Invalid info (parent)\n", iteration);
	   }

	   if((MPI_ERR_PROC_FAILED == ret) || (MPI_ERR_REVOKED == ret)){
	      OMPI_Comm_revoke(tempShrink);
              return repairComm(broken, repaired, iteration, listFails, numFails, numNodeFails,
                                sumPrevNumNodeFails, argc, argv, verbosity);
	   }
	   else{
	      fprintf(stderr, "Iteration %d: Unknown error with MPI_Comm_spawn_multiple (parent): %d\n", iteration, ret);
	      exit(1);
	   }
	}
	else{
	   if(verbosity > 0 && rank == 0){
	      for(i = 0; i < procsNeeded; i++){
		 printf("Iteration %d: MPI_Comm_spawn_multiple (parent) [spawning failed process %d on "
                        "node %s]: SUCCESS\n", iteration, failedRanks[i], hostNameToLaunch[i]);
	      }
	   }
	   // Memory release. Moving the last two to the end of the function causes segmentation faults for 4 processes failure
	}
	if (verbosity > 0 && rank == 0){
	   printf("MPI_Comm_spawn_multiple takes %0.6f Sec\n", MPI_Wtime() - spawnTime);
	}

	double mergeTime = MPI_Wtime();
	// Merge the new processes into a new communicator
	if(MPI_SUCCESS != (ret = MPI_Intercomm_merge(tempIntercomm, false, &unorderIntracomm))){
	   free(tempRanks);
	   free(failedRanks);
	   if((MPI_ERR_PROC_FAILED == ret) || (MPI_ERR_REVOKED == ret)){
	      // Start the recovery over again if there is a failure
	      OMPI_Comm_revoke(tempIntercomm);
              return repairComm(broken, repaired, iteration, listFails, numFails,
                                numNodeFails, sumPrevNumNodeFails, argc, argv, verbosity);
	   }
	   else if(MPI_ERR_COMM == ret){
	      fprintf(stderr, "Iteration %d: Invalid communicator in MPI_Intercomm_merge (parent) %d\n", iteration, ret);
	      exit(1);
	   }
	   else if(MPI_ERR_INTERN == ret){
	      fprintf(stderr, "Iteration %d: Acquaring memory error in MPI_Intercomm_merge ()%d\n", iteration, ret);
	      exit(1);
	   }
	   else{
	      fprintf(stderr, "Iteration %d: Unknown error with MPI_Intercomm_merge: %d\n", iteration, ret);
	      exit(1);
	   }
	}
	else{
	   if(verbosity > 1 ){
	      printf("Iteration %d: MPI_Intercomm_merge (parent): SUCCESS\n", iteration);
	   }
	}
	if (verbosity > 0 && rank == 0){
	   printf("MPI_Intercomm_merge takes %0.6f Sec\n", MPI_Wtime() - mergeTime);
	}

        double agreeTime = MPI_Wtime();
	// Synchronize. sometimes hangs in without this
	// position of code and intercommunicator (not intra) is important
#ifdef HANG_ON_REMOVE
        //MPI_Barrier(tempIntercomm);
        OMPI_Comm_agree(tempIntercomm, &flag);// since some of the times MPI_Barrier hangs 
#endif
	if (verbosity > 0 && rank == 0){
	   printf("OMPI_Comm_agree takes %0.6f Sec\n", MPI_Wtime() - agreeTime);
	}

	// Sending failed ranks and number of processes failed to the the newly created ranks.
	// oldGroupSize is the size of communicator before failure.
	// procsNeeded is the number of processes that are failed
	int * child = (int *) malloc(procsNeeded*sizeof(int));
	#pragma omp parallel for default(shared)
	for(i = 0; i < procsNeeded; i++){
	   child[i] = oldGroupSize - procsNeeded + i;
	}

	MPI_Comm_rank(unorderIntracomm, &newRank);
	if(newRank == 0){
	   int send_val[2];
	   for(i = 0; i < procsNeeded; i++){
	      send_val[0] = failedRanks[i]; send_val[1] = procsNeeded;
	      if(MPI_SUCCESS != (ret = MPI_Send(&send_val, 2, MPI_INT, child[i], MERGE_TAG, unorderIntracomm))){
		 free(tempRanks);
		 free(failedRanks);
		 if((MPI_ERR_PROC_FAILED == ret) || (MPI_ERR_REVOKED == ret)){
		    // Start the recovery over again if there is a failure
		    OMPI_Comm_revoke(unorderIntracomm);
                    return repairComm(broken, repaired, iteration, listFails, numFails,
                                      numNodeFails, sumPrevNumNodeFails, argc, argv, verbosity);
		 }
		 else{
		    fprintf(stderr, "Iteration %d: Unknown error with MPI_Send1 (parent): %d\n", iteration, ret);
		    exit(1);
		 }
	      }
	      else{
		 if(verbosity > 1 ){
		    printf("Iteration %d: MPI_Send1 (parent): SUCCESS\n", iteration);
		 }
	      }
	   }
	}

	// Split the current world (splitted from original) to order the ranks.
	MPI_Comm_rank(unorderIntracomm, &newRank);
	MPI_Comm_size(unorderIntracomm, &nprocs);

	// For one or more process failure (ordering)
	shrinkMergeList = (int *) malloc(nprocs*sizeof(int));

	j = 0;
	for(i = 0; i < nprocs; i++){
	   if(rankIsNotOnFailedList(i, failedRanks, procsNeeded)){
	      shrinkMergeList[j++] = i;
	   }
	}

	for(i = j; i < nprocs; i++){
	   shrinkMergeList[i] = failedRanks[i-j];
	}

	for(i = 0; i < (nprocs - procsNeeded); i++){
	   if(newRank == i){
	      rankKey = shrinkMergeList[i];
	   }
	}

	if(MPI_SUCCESS != (MPI_Comm_split(unorderIntracomm, 0, rankKey, repaired))){
	   if((MPI_ERR_PROC_FAILED == ret) || (MPI_ERR_REVOKED == ret)){
	      // Start the recovery over again if there is a failure
	      OMPI_Comm_revoke(unorderIntracomm);
              return repairComm(broken, repaired, iteration, listFails, numFails,
                                numNodeFails, sumPrevNumNodeFails, argc, argv, verbosity);
	   }
	   else{
	      fprintf(stderr, "Iteration %d: Unknown error with MPI_Comm_split (parent): %d\n", iteration, ret);
	      exit(1);
	   }
	}
	else{
	   if(verbosity > 1 ){
	      printf("Iteration %d: MPI_Comm_split (parent): SUCCESS\n", iteration);
	   }
	}

	// Release memory
	free(appToLaunch);
	free(argvToLaunch);
	free(procsNeededToLaunch);
	free(hostInfoToLaunch);
	free(hostNameToLaunch);
	free(shrinkMergeList);
	free(errCodes);
	MPI_Comm_free(&tempShrink);
	free(tempRanks);
	free(failedRanks);
	free(child);
	MPI_Group_free(&failedGroup);
	MPI_Group_free(&oldGroup);
	MPI_Comm_free(&tempIntercomm);
        MPI_Comm_free(&unorderIntracomm);        
#ifdef RECOV_ON_SPARE_NODES
        if(failedNodeList != NULL){
           free(failedNodeList);
        }
       if(nodeList != NULL){
           free(nodeList);
        }
#endif      
}//repairComm()


////////////////////////////////////////////////////////////////////
int rankIsNotOnFailedList(int rank, int * failedList, int numFails){
     int i;

     for(i = 0; i < numFails; i++){
	if(rank == failedList[i]){
	   return 0;
	}
     }

     if(i == numFails){
	return 1;
     }

     return -1; // error
}//rankIsNotOnFailedList()

/////////////////////////////////////////////////////////////////////////////////////////
int getHostfileLastLineIndex(void){
     // Get the code from http://stackoverflow.com/questions/10476503/how-can-i-select-the-last-line-of-a-text-file-using-c
     // hostfile line index is started from 0
     FILE *fPointer;
     if(NULL == (fPointer = fopen("hostfile", "r"))){
	 printf("Error in opening file. Exit\n");
	 exit(0);
     }

     int lastLineIndex = 0;
     char *buffer = (char *) malloc(1024);

     fseek(fPointer, 0, SEEK_SET); // make sure start from 0
     while(!feof(fPointer)){
        memset(buffer, 0x00, 1024); // clean buffer
        int ret = fscanf(fPointer, "%[^\n]\n", buffer); // read file * prefer using fscanf
        lastLineIndex++;
        if(ret == EOF){
           break;
        }
     }

     // Memory deallocation
     fclose(fPointer);

     if(lastLineIndex - 1 == -1) {
        printf("hotfile is empty...\n");
        fflush(NULL);
        exit(0);
     }
     else {
        return (lastLineIndex - 1);
     }
}//getHostfileLastLineIndex()

/////////////////////////////////////////////////////////////////////////////////////////
char * getHostToLaunch(int hostfileLineIndex){
     FILE *fPointer;
     if(NULL == (fPointer = fopen("hostfile", "r"))){
	 printf("Error in opening file. Exit\n");
	 exit(0);
     }

     char lineRead[256];
     char * buffer;
     int curLineIndex = 0;

     // extract a line of string from hostfile "hostfile" (with hostname followed by slots)
     // with line index "hostfileLineIndex" (started from 0)
     while(curLineIndex <= hostfileLineIndex){
	 if(NULL == (fgets(lineRead, 256, fPointer))){
	     printf("fgets() encountered NULL\nProblem in hostfile\nExit\n");
	     exit(0);
	 }
	 curLineIndex++;
     }

     buffer = (char *) malloc(strlen(lineRead)+1);
     strcpy(buffer, lineRead);

     //printf("Line that read is: %s\n", buffer);

     // extract only hostname without slots information
     buffer = strtok (buffer, " ");

     // memory deallocation
     fclose(fPointer);
     // buffer is deallocated from caller side

     return buffer;
}//getHostToLaunch()


//////////////////////////////////////////////////////////////////////////////////
int ranksOnSameGrid(int rank, int numFails, int * listFails, GridCombine2D * gc) {
    int i;
    for(i = 0; i < numFails; i++){
       if(gc->getGid(rank) == gc->getGid(listFails[i])){
          return 1;
       }
    }
    if(i == numFails){
       return 0;
    }

    return -1; // Unknown error
}//ranksOnSameGrid()


/////////////////////////////////////////////////////////////////////////////
int isPreviouslySend(int currentIndex, int * listFails, GridCombine2D * gc) {
    int i;
    for(i = 0; i < currentIndex; i++){
       if(gc->getGid(listFails[currentIndex]) == gc->getGid(listFails[i])){
	  return 1;
       }
    }
    if(i == currentIndex){
       return 0;
    }

    return -1; // Unknown error
}//isPreviouslySend()


/////////////////////////////////////////////////////////////////////////////
int minRank(int fGid, int nprocs, GridCombine2D * gc) {
    int i, min = 2*nprocs;
    for(i = 0; i < nprocs; i++){
       if((fGid == gc->getGid(i)) && (i < min)){
	  min = i;
       }
    }

    return min;
}//minRank()


//////////////////////////////////////////////////////////////////////
int isRightBoundary(Vec2D<int> id, Vec2D<int> P) {
    return (P.x-1 == id.x);
}//isRightBoundary()

//////////////////////////////////////////////////////////////////////
int isUpperBoundary(Vec2D<int> id, Vec2D<int> P) {
    return (P.y-1 == id.y);
}//isUpperBoundary()

//////////////////////////////////////////////////////////////////////
int getToRank(int min_rank, int fidx, int fidy, int fPx) {
    return (min_rank + fidx + fidy * fPx);
}//getToRank()

/////////////////////////////////////////////////////////////////////
// twoPower function: allocate and initialize with 0.0 //
double twoPow(int x){
    return pow((double)2, (double)x);
}//twoPower()

/////////////////////////////////////////////////////////////////////
double ** alloc2dDouble(int rows, int cols){
    double * data = (double *) malloc(rows*cols*sizeof(double));
    double ** array = (double **)malloc(rows*sizeof(double *));

    int i, j;

    for(i = 0; i < rows; i++){
       array[i] = &(data[cols*i]);
    }

    // initialize
    #pragma omp parallel for default(shared)
    for(j = 0; j < cols; j++){
       for(i = 0; i < rows; i++){
	  array[i][j] = 0.0;
       }
    }

    return array;
}//alloc2dDouble()

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
void recoveryByResampling(HaloArray2D *u, ProcGrid2D *g, MPI_Comm myCommWorld, int isParent, GridCombine2D *gc, 
                         int rank, int nprocs, int level, int numFails, int * listFails){
    // Constraints:
    // (1) If failure is on middle grid, then interpolation (by resampling) from its upper neighbor grid
    // (2) a<=c, b<=d for (1), where a = g->fP.x and b = g->fP.y (failed process grid's dimensions)
    // (3) If failure is on upper or lower grid, then copying the same recomputed values
    // (4) No of processes on non-failed grid is the same as no of processes on failed grid for (3)

    int i, j, m, tempP[3], nfGridId = 0, gridId, fGridId, nfPxyId[3];
    MPI_Status nonFailedStatus, nfProcsGridStatus, nfProcsGridStatus2;

    // previously failed and spawned
    if(isParent == 0){
       fGridId = gc->getGid(rank);
       int min_interpolate_rank = 2*nprocs;

       for(i = 0; i < nprocs; i++){
	  // failed process sends P.x and P.y to neighbor processes from where interpolation should happen
	  gridId = gc->getGid(i);
	  if((fGridId < (level-1) && fGridId > 0 && gridId == (fGridId + 2*(level-1))) ||
		(fGridId == 0 && gridId == (3*level - 3)) ||
		(fGridId == (3*level - 3) && gridId == 0) ||
		(fGridId == (level-1) && gridId == (3*level - 2)) ||
		(fGridId == (3*level-2) && gridId == (level-1)) ||
		(fGridId >= level && fGridId < (2*level - 1) && gridId == (fGridId - (level-1))) ||
		(fGridId >= (2*level - 1) && fGridId < (3*level - 3) && gridId == (fGridId - 2*(level-1)))){
	     //printf("[%d] Sending (%d, %d) to process %d complete\n", rank, tempP[0], tempP[1], i);
	     if(i < min_interpolate_rank){
		// process with minimum rank in a grid, that is, 2D rank in processor grid is (0,0)
		min_interpolate_rank = i;
	     }
	  } // end of if
       } // end of for

       // failed process receives P.x, P.y, and gridId from neighbor process (min_interpolate_rank) from where interpolation should happen
       MPI_Recv(&tempP, 3, MPI_INT, min_interpolate_rank, PROCS_GRID_NF_TAG, myCommWorld, &nfProcsGridStatus);
       //printf("                          [Failed process %d] Receiving from  %d\n", rank, min_interpolate_rank);
       nfPxyId[0] = tempP[0]; nfPxyId[1] = tempP[1]; nfPxyId[2] = tempP[2];

       int pGridSize = (g->P.x)*(g->P.y);
       int minIndex = ((rank - pGridSize) >= 0)?(rank - pGridSize):0;
       int maxIndex = ((rank + pGridSize) <= nprocs)?(rank + pGridSize):nprocs;
       // failed process sends P.x, P.y, and gridId to other processes that are on its own grid
       for(i = minIndex; i < maxIndex; i++){
	  if(fGridId == gc->getGid(i) && rank != i){
	     MPI_Send(&nfPxyId, 3, MPI_INT, i, PROCS_GRID_NF_TAG2, myCommWorld);
	     //printf("                                        [Failed process %d] Sending to same-grid process %d\n", rank, i);
	  }
       }
    }// end of if
    // this is child's same-grid process
    if(isParent == 0 || (isParent != 0 && ranksOnSameGrid(rank, numFails, listFails, gc))){
       for(i = 0; i < numFails; i++){
	  int tempList = listFails[i];
	  if((rank != tempList) && (gc->getGid(rank) == gc->getGid(tempList))){
	     // processes that are on the same grid as failed grid receive P.x, P.y, and gridId list from failed process
	     MPI_Recv(&nfPxyId, 3, MPI_INT, tempList, PROCS_GRID_NF_TAG2, myCommWorld, &nfProcsGridStatus2);
	     //printf("                                                                 [Same-grid process %d] Receiving from failed process %d\n", rank, tempList);
	  }
       }
    }// end of else if
    // not spawned
    else if(isParent != 0){
	for(m = 0; m < numFails; m++){
	   int fGridId = gc->getGid(listFails[m]);
	   gridId = gc->getGid(rank);
	   Vec2D<int> fGrid = gc->gridIx(fGridId);
	      // neighbor processes from where interpolation should happen receive P.x and P.y from failed process
	      if((fGridId < (level-1) && fGridId > 0 && gridId == (fGridId + 2*(level-1))) ||
		       (fGridId == 0 && gridId == (3*level - 3)) ||
		       (fGridId == (3*level - 3) && gridId == 0) ||
		       (fGridId == (level-1) && gridId == (3*level - 2)) ||
		       (fGridId == (3*level-2) && gridId == (level-1)) ||
		       (fGridId >= level && fGridId < (2*level - 1) && gridId == (fGridId - (level-1))) ||
		       (fGridId >= (2*level - 1) && fGridId < (3*level - 3) && gridId == (fGridId - 2*(level-1)))){
		 ProcGrid2D *fg = gc->getPgs(fGridId);

		 // neighbor process (min_interpolate_rank) from where interpolation should happen sends P.x, P.y, and gridId to failed process
		 if(g->id.x == 0 && g->id.y == 0){
		    tempP[0] = g->P.x; tempP[1] = g->P.y; tempP[2] = gridId;
		    MPI_Send(&tempP, 3, MPI_INT, listFails[m], PROCS_GRID_NF_TAG, myCommWorld);
		    //printf("          [%d] Sending to failed process %d\n", rank, listFails[m]);
		 }

		 int c = g->P.x, d = g->P.y, k = g->id.x, l = g->id.y;
		 int a = fg->P.x, b = fg->P.y;
		 int right, upper;
		 int startX, endX, startY, endY, denomEndX = 1, denomEndY = 1;
		 int sk, si, indexX = 0, indexY = 0, destination, nLx, nLy;
		 double lastIndexX, lastIndexY, nLastIndexX = 0.0, nLastIndexY = 0.0;
		 double ** buff = NULL;
		 int yIncrFlag = 0;

		 right = (a>c)?a/c:c/a; upper = (b>d)?b/d:d/b;

		 if(!(isRightBoundary(g->id, g->P) || isUpperBoundary(g->id, g->P))){
		    //printf("Not right, not upper: rank %d, i %d, j %d, k %d, l %d, right %d, upper %d\n", rank, i, j, k, l, right, upper);
		    denomEndX = (a>c)?a/c:1; denomEndY = (b>d)?b/d:1;
		    nLx = u->l.x; nLy = u->l.y;
		 }
		 if(isRightBoundary(g->id, g->P) && !isUpperBoundary(g->id, g->P)){
		    //printf("Right, not upper: rank %d, i %d, j %d, k %d, l %d, right %d, upper %d\n", rank, i, j, k, l, right, upper);
		    denomEndX = (a>c)?(a/c + a%c):1; denomEndY = (b>d)?b/d:1;
		    nLx = u->l.x-1; nLy = u->l.y;
		 }
		 if(!isRightBoundary(g->id, g->P) && isUpperBoundary(g->id, g->P)){
		    //printf("Not right, upper: rank %d, i %d, j %d, k %d, l %d, right %d, upper %d\n", rank, i, j, k, l, right, upper);
		    denomEndX = (a>c)?a/c:1; denomEndY = (b>d)?(b/d + b%d):1;
		    nLx = u->l.x; nLy = u->l.y-1;
		 }
		 if(isRightBoundary(g->id, g->P) && isUpperBoundary(g->id, g->P)){
		    //printf("Right and upper: rank %d, i %d, j %d, k %d, l %d, right %d, upper %d\n", rank, i, j, k, l, right, upper);
		    denomEndX = (a>c)?(a/c + a%c):1; denomEndY = (b>d)?(b/d + b%d):1;
		    nLx = u->l.x-1; nLy = u->l.y-1;
		 }

		 lastIndexX = twoPow(fGrid.x)/(double)a;
		 lastIndexY = twoPow(fGrid.y)/b;

		 startY = 1; endY = ((startY+(nLy/denomEndY)));
		 for(j = 0; j < fg->P.y; j++){ // j is fidy
		    startX = 1; endX = ((startX+(nLx/denomEndX)));
		    for(i = 0; i < fg->P.x; i++){ // i is fidx
		       // select group of process' grid coordinate who take part sending their portions to the mapped process on failed grid
		       if(((((a>=c)?i:k) < (std::min(a, c)*right)) && (((a>=c)?k:i) == (((a>=c)?i:k)/right)) && (((b>=d)?j:l) < 
			       (std::min(b, d)*upper)) &&
			       (((b>=d)?l:j) == (((b>=d)?j:l)/upper))) || ((((a>=c)?i:k) >= (std::min(a, c)*right)) && 
						(((a>=c)?k:i) == (((a>=c)?i:k)/(right+1))) &&
			       (((b>=d)?j:l) < (std::min(b, d)*upper)) && (((b>=d)?l:j) == (((b>=d)?j:l)/upper))) || ((((a>=c)?i:k) < 
				 (std::min(a, c)*right)) &&
			       (((a>=c)?k:i) == (((a>=c)?i:k)/right)) && (((b>=d)?j:l) >= (std::min(b, d)*upper)) &&
			       (((b>=d)?l:j) == (((b>=d)?j:l)/(upper+1)))) || ((((a>=c)?i:k) >= (std::min(a, c)*right)) &&
			       (((a>=c)?k:i) == (((a>=c)?i:k)/(right+1))) && (((b>=d)?j:l) >= (std::min(b, d)*upper)) && (((b>=d)?l:k) == 
			       (((b>=d)?j:l)/(upper+1))))){
			  if(!isPreviouslySend(m, listFails, gc)){
			      sk = startY; si = startX;
			      yIncrFlag = 1;

			      if(i != (a-1) && j != (b-1)){// allocate and initialize each with 0.0
				 nLastIndexX = lastIndexX;
				 nLastIndexY = lastIndexY;
				 buff = alloc2dDouble((int)nLastIndexX+2, (int)nLastIndexY+2);
			      }
			      if(i == (a-1) && j != (b-1)){
				 nLastIndexX = lastIndexX + 1;
				 nLastIndexY = lastIndexY;
				 buff = alloc2dDouble((int)nLastIndexX+2, (int)nLastIndexY+2);
			      }
			      if(i != (a-1) && j == (b-1)){
				 nLastIndexX = lastIndexX;
				 nLastIndexY = lastIndexY + 1;
				 buff = alloc2dDouble((int)nLastIndexX+2, (int)nLastIndexY+2);
			      }
			      if(i == (a-1) && j == (b-1)){
				 nLastIndexX = lastIndexX + 1;
				 nLastIndexY = lastIndexY + 1;
				 buff = alloc2dDouble((int)nLastIndexX+2, (int)nLastIndexY+2);
			      }

			      int xDiv = (int)(c/a);
			      int yDiv = (int)(d/b);
			      int idFactorX = (g->id.x)%xDiv;
			      int idFactorY = (g->id.y)%yDiv;
			      int diffEndStartX = endX-startX;
			      int diffEndStartY = endY-startY;

			      Vec2D<int> nfGrid = gc->gridIx(gridId);
			      double xSelect = twoPow(fGrid.x)/twoPow(nfGrid.x);
			      double ySelect = twoPow(fGrid.y)/twoPow(nfGrid.y);

			      // middle grid
			      if(fGridId >= level && fGridId < (2*level-1)){
				  if((xDiv > 1.0) && (yDiv > 1.0)){
				      //printf("*** [%d] I am here: xfactor = %0.2f, yfactor = %0.2f, xSelect = %0.2f, ySelect = %0.2f\n", 
				      //rank, xfactor, yfactor, xSelect, ySelect);;

				      for(indexY = 0; indexY < (int)nLastIndexY; indexY++){
					for(indexX = 0; indexX < (int)nLastIndexX; indexX++){

					   if((idFactorX == 0 && (int)(indexX/xSelect) >= diffEndStartX) || (idFactorY == 0 && 
					      (int)(indexY/ySelect) >= diffEndStartY)){
					      buff[indexX+1][indexY+1] = -99.0;
					   }
					   else if((idFactorX != 0 && idFactorX != (xDiv-1) && (((int)(indexX/xSelect) < idFactorX*diffEndStartX) 
						   || ((int)(indexX/xSelect) >= (idFactorX+1)*diffEndStartX))) ||
						   (idFactorY != 0 && idFactorY != (yDiv-1) && (((int)(indexY/ySelect) < idFactorY*diffEndStartY) 
						    || ((int)(indexY/ySelect) >= (idFactorY+1)*diffEndStartY)))){
					      buff[indexX+1][indexY+1] = -99.0;
					   }
					   else if((idFactorX == (xDiv-1) && (int)(indexX/xSelect) < idFactorX*diffEndStartX) || (idFactorY == 
						    (yDiv-1) && (int)(indexY/ySelect) < idFactorY*diffEndStartY)){
					      buff[indexX+1][indexY+1] = -99.0;
					   }
					   else if(idFactorX == 0 && (int)(indexX/xSelect) < diffEndStartX){
					       if(idFactorY == 0 && (int)(indexY/ySelect) < diffEndStartY){
						  buff[indexX+1][indexY+1] = V(u, si+(int)(indexX/xSelect), sk+(int)(indexY/ySelect));
					       }
					       else if(idFactorY != 0 && idFactorY != (yDiv-1) && (((int)(indexY/ySelect) >= idFactorY*
						       diffEndStartY) && ((int)(indexY/ySelect) < (idFactorY+1)*diffEndStartY))){
						  buff[indexX+1][indexY+1] = V(u, si+(int)(indexX/xSelect), sk+(int)((int)(indexY/ySelect)-
									       idFactorY*diffEndStartY));
					       }
					       else if(idFactorY == (yDiv-1) && (int)(indexY/ySelect) >= idFactorY*diffEndStartY){
						  buff[indexX+1][indexY+1] = V(u, si+(int)(indexX/xSelect), sk+(int)((int)(indexY/ySelect)-
									       idFactorY*diffEndStartY));
					       }
					   }
					   else if(idFactorX != 0 && idFactorX != (xDiv-1) && (((int)(indexX/xSelect) >= idFactorX*diffEndStartX) 
						   && ((int)(indexX/xSelect) < (idFactorX+1)*diffEndStartX))){
					       if(idFactorY == 0 && (int)(indexY/ySelect) < diffEndStartY){
						  buff[indexX+1][indexY+1] = V(u, si+(int)((int)(indexX/xSelect)-idFactorX*diffEndStartX), 
									       sk+(int)(indexY/ySelect));
					       }
					       else if(idFactorY != 0 && idFactorY != (yDiv-1) && (((int)(indexY/ySelect) >= 
							     idFactorY*diffEndStartY) && ((int)(indexY/ySelect) < (idFactorY+1)*diffEndStartY))){
						  buff[indexX+1][indexY+1] = V(u, si+(int)((int)(indexX/xSelect)-idFactorX*diffEndStartX), 
									       sk+(int)((int)(indexY/ySelect)-idFactorY*diffEndStartY));
					       }
					       else if(idFactorY == (yDiv-1) && (int)(indexY/ySelect) >= idFactorY*diffEndStartY){
						  buff[indexX+1][indexY+1] = V(u, si+(int)((int)(indexX/xSelect)-idFactorX*diffEndStartX), 
									       sk+(int)((int)(indexY/ySelect)-idFactorY*diffEndStartY));
					       }
					   }
					   else if(idFactorX == (xDiv-1) && (int)(indexX/xSelect) >= idFactorX*diffEndStartX){
					       if(idFactorY == 0 && (int)(indexY/ySelect) < diffEndStartY){
						  buff[indexX+1][indexY+1] = V(u, si+(int)((int)(indexX/xSelect)-idFactorX*diffEndStartX), 
									       sk+(int)(indexY/ySelect));
					       }
					       else if(idFactorY != 0 && idFactorY != (yDiv-1) && (((int)(indexY/ySelect) >= 
						       idFactorY*diffEndStartY) && ((int)(indexY/ySelect) < (idFactorY+1)*diffEndStartY))){
						  buff[indexX+1][indexY+1] = V(u, si+(int)((int)(indexX/xSelect)-idFactorX*diffEndStartX), 
									       sk+(int)((int)(indexY/ySelect)-idFactorY*diffEndStartY));
					       }
					       else if(idFactorY == (yDiv-1) && (int)(indexY/ySelect) >= idFactorY*diffEndStartY){
						  buff[indexX+1][indexY+1] = V(u, si+(int)((int)(indexX/xSelect)-idFactorX*diffEndStartX), 
									       sk+(int)((int)(indexY/ySelect)-idFactorY*diffEndStartY));
					       }
					   }
					   else{
					       printf("Problem on IF ELSE 1\n");
					   }
					   //printf("%d(%d,%d)%0.2f ", rank, indexX+1, indexY+1, buff[indexX+1][indexY+1]);
					}
					//printf("\n");
				     }
				  }
				  if((xDiv <= 1.0) && (yDiv <= 1.0)){
				      //printf("*** [%d] I am here: xfactor = %0.2f, yfactor = %0.2f, xSelect = %0.2f, ySelect = %0.2f\n", 
				      //rank, xfactor, yfactor, xSelect, ySelect);;

				      for(indexY = 0; indexY < (int)nLastIndexY; indexY++){
					for(indexX = 0; indexX < (int)nLastIndexX; indexX++){
					   if((idFactorX == 0 && (int)(indexX/xSelect) >= diffEndStartX) || (idFactorY == 0 && 
					     (int)(indexY/ySelect) >= diffEndStartY)){
					      buff[indexX+1][indexY+1] = -99.0;
					   }
					   else if((idFactorX == 0 && (int)(indexX/xSelect) < diffEndStartX) && (idFactorY == 0 && 
						  (int)(indexY/ySelect) < diffEndStartY)){
					       buff[indexX+1][indexY+1] = V(u, si+(int)(indexX/xSelect), sk+(int)(indexY/ySelect));
					   }
					   else{
					       printf("Problem on IF ELSE 2\n");
					   }
					   //printf("%d(%d,%d)%0.2f ", rank, indexX+1, indexY+1, buff[indexX+1][indexY+1]);
					}
					//printf("\n");
				     }
				  }
				  if((xDiv > 1.0) && (yDiv <= 1.0)){
				      //printf("*** [%d] I am here: xfactor = %0.2f, yfactor = %0.2f, xSelect = %0.2f, ySelect = %0.2f\n", 
				      //rank, xfactor, yfactor, xSelect, ySelect);

				      for(indexY = 0; indexY < (int)nLastIndexY; indexY++){
					for(indexX = 0; indexX < (int)nLastIndexX; indexX++){
					   if(idFactorX == 0 && (int)(indexX/xSelect) >= diffEndStartX){
					      buff[indexX+1][indexY+1] = -99.0;
					   }
					   else if(idFactorX != 0 && idFactorX != (xDiv-1) && (((int)(indexX/xSelect) < idFactorX*diffEndStartX) ||                                               ((int)(indexX/xSelect) >= (idFactorX+1)*diffEndStartX))){
					      buff[indexX+1][indexY+1] = -99.0;
					   }
					   else if(idFactorX == (xDiv-1) && (int)(indexX/xSelect) < idFactorX*diffEndStartX){
					      buff[indexX+1][indexY+1] = -99.0;
					   }
					   else if(idFactorX == 0 && (int)(indexX/xSelect) < diffEndStartX){
					      buff[indexX+1][indexY+1] = V(u, si+(int)(indexX/xSelect), sk+(int)(indexY/ySelect));
					   }
					   else if(idFactorX != 0 && idFactorX != (xDiv-1) && (((int)(indexX/xSelect) >= idFactorX*diffEndStartX)                                                && ((int)(indexX/xSelect) < (idFactorX+1)*diffEndStartX))){
					      buff[indexX+1][indexY+1] = V(u, si+(int)((int)(indexX/xSelect)-idFactorX*diffEndStartX), 
									 sk+(int)(indexY/ySelect));
					   }
					   else if(idFactorX == (xDiv-1) && (int)(indexX/xSelect) >= idFactorX*diffEndStartX){
					      buff[indexX+1][indexY+1] = V(u, si+(int)((int)(indexX/xSelect)-idFactorX*diffEndStartX), 
									 sk+(int)(indexY/ySelect));
					   }
					   else{
					       printf("Problem on IF ELSE 3\n");
					   }
					   //printf("%d(%d,%d)%0.2f ", rank, indexX+1, indexY+1, buff[indexX+1][indexY+1]);
					}
					//printf("\n");
				     }
				  }

				  if((xDiv <= 1.0) && (yDiv > 1.0)){
				     //printf("*** [%d] I am here: xfactor = %0.2f, yfactor = %0.2f, lastIndexX = %d, nLastIndexX = %d, endX = %d,
				     // startX = %d, lastIndexY = %d, nLastIndexY = %d, endY = %d, startY = %d\n", rank, xfactor, yfactor, 
				     //(int)lastIndexX, (int)nLastIndexX, endX, startX, (int)lastIndexY, (int)nLastIndexY, endY, startY);
				     for(indexY = 0; indexY < (int)nLastIndexY; indexY++){
					for(indexX = 0; indexX < (int)nLastIndexX; indexX++){
					   if(idFactorY == 0 && (int)(indexY/ySelect) >= diffEndStartY){
					      buff[indexX+1][indexY+1] = -99.0;
					   }
					   else if(idFactorY != 0 && idFactorY != (yDiv-1) && (((int)(indexY/ySelect) < idFactorY*diffEndStartY) ||                                              ((int)(indexY/ySelect) >= (idFactorY+1)*diffEndStartY))){
					      buff[indexX+1][indexY+1] = -99.0;
					   }
					   else if(idFactorY == (yDiv-1) && (int)(indexY/ySelect) < idFactorY*diffEndStartY){
					      buff[indexX+1][indexY+1] = -99.0;
					   }
					   else if(idFactorY == 0 && (int)(indexY/ySelect) < diffEndStartY){
					      buff[indexX+1][indexY+1] = V(u, si+(int)(indexX/xSelect), sk+(int)(indexY/ySelect));
					   }
					   else if(idFactorY != 0 && idFactorY != (yDiv-1) && (((int)(indexY/ySelect) >= idFactorY*diffEndStartY) 
						   && ((int)(indexY/ySelect) < (idFactorY+1)*diffEndStartY))){
					      buff[indexX+1][indexY+1] = V(u, si+(int)(indexX/xSelect), sk+(int)((int)(indexY/ySelect)-
									   idFactorY*diffEndStartY));
					   }
					   else if(idFactorY == (yDiv-1) && (int)(indexY/ySelect) >= idFactorY*diffEndStartY){
					      buff[indexX+1][indexY+1] = V(u, si+(int)(indexX/xSelect), sk+(int)((int)(indexY/ySelect)-
									   idFactorY*diffEndStartY));
					   }
					   else{
					       printf("Problem on IF ELSE 4\n");
					   }
					   //printf("%d(%d,%d)%0.2f ", rank, indexX+1, indexY+1, buff[indexX+1][indexY+1]);
					}
					//printf("\n");
				     }
				  }
			      } // end of if "middle grid"
			      // grid with id 0 to (level-1), (2*level-1) to (3*level-2)
			      else if((fGridId < level && fGridId >= 0) || (fGridId >= (2*level-1) && fGridId <= (3*level-2))){
				  for(indexY = 0; indexY < (int)nLastIndexY; indexY++){
				    for(indexX = 0; indexX < (int)nLastIndexX; indexX++){
					buff[indexX+1][indexY+1] = V(u, si+indexX, sk+indexY);
				       //printf("%d(%d,%d)%0.2f ", rank, indexX+1, indexY+1, buff[indexX+1][indexY+1]);
				    }
				    //printf("\n");
				 }
			      } // end of else if "grid with id 0 to (level-1), (2*level-1) to (3*level-2)"

			      destination = getToRank(minRank(fGridId, nprocs, gc), i, j, a);
			      //printf("     [%d] Send to %d\n", rank, destination);

			      MPI_Send(&(buff[0][0]), (int)(nLastIndexX+2)*(int)(nLastIndexY+2), MPI_DOUBLE, destination, 
				       NON_FAILED_FAILED_TAG, myCommWorld);
			      // free the allocated memory for buff
			      free(buff[0]);
			      free(buff);

			      startX = endX;
			      endX = ((startX+((u->s.x-2)/denomEndX)));
			  }// end of "if(!isPreviouslySend("
		       }// end of "if(((((k < ..."
		    }// end of "for(i...)"

		    if(yIncrFlag == 1){
		       startY = endY;
		       endY = ((startY+((u->s.y-2)/denomEndY)));

		       yIncrFlag = 0;
		    }
		 }// end of "for(j...)"
	      }// end of "if((fGridId ..."
	  }// end of "for(i = 0;...)
       }// end of else

       // this is child or child's same-grid process
       // listFails is NOT available to recovered process(es)
       if(isParent == 0 || (isParent != 0 && ranksOnSameGrid(rank, numFails, listFails, gc))){ //printf("                                                           isParent == 0 || SameGrid 2: rank = %d\n", rank);
	  int c = g->P.x, d = g->P.y, k = g->id.x, l = g->id.y;
	  int a, b, right, upper, ri, rj, source;

	  fGridId = gc->getGid(rank);

	  a = nfPxyId[0]; b = nfPxyId[1];  nfGridId = nfPxyId[2];

	  for(j = 0; j < b; j++){ // j is fidy
	     for(i = 0; i < a; i++){ // i is fidx
		right = (a>c)?a/c:c/a; upper = (b>d)?b/d:d/b;

		if(((((a>=c)?i:k) < (std::min(a, c)*right)) && (((a>=c)?k:i) == (((a>=c)?i:k)/right)) &&
			   (((b>=d)?j:l) < (std::min(b, d)*upper)) && (((b>=d)?l:j) == (((b>=d)?j:l)/upper))) ||
			   ((((a>=c)?i:k) >= (std::min(a, c)*right)) && (((a>=c)?k:i) == (((a>=c)?i:k)/(right+1))) &&
			   (((b>=d)?j:l) < (std::min(b, d)*upper)) && (((b>=d)?l:j) == (((b>=d)?j:l)/upper))) ||
			   ((((a>=c)?i:k) < (std::min(a, c)*right)) && (((a>=c)?k:i) == (((a>=c)?i:k)/right)) &&
			   (((b>=d)?j:l) >= (std::min(b, d)*upper)) && (((b>=d)?l:j) == (((b>=d)?j:l)/(upper+1)))) || 
			  ((((a>=c)?i:k) >= (std::min(a, c)*right)) && (((a>=c)?k:i) == (((a>=c)?i:k)/(right+1))) && 
			  (((b>=d)?j:l) >= (std::min(b, d)*upper)) && (((b>=d)?l:k) == (((b>=d)?j:l)/(upper+1))))){

		   source = getToRank(minRank(nfGridId, nprocs, gc), i, j, a);
		   //printf("                [%d] Receive from %d\n", rank, source);
		   double ** recvBuff = alloc2dDouble(u->s.x, u->s.y);

		   MPI_Recv(&(recvBuff[0][0]), (u->s.x)*(u->s.y), MPI_DOUBLE, source, NON_FAILED_FAILED_TAG, myCommWorld, &nonFailedStatus);

		   for(rj = 0; rj < u->s.y; rj++){
		      for(ri = 0; ri < u->s.x; ri++){
			 if(recvBuff[ri][rj] != -99.0){
			    *(u->ix(ri, rj)) = recvBuff[ri][rj];
			 }
			 //if(rank == 25)printf("%d(%d,%d)%0.2f ", source, ri, rj,  recvBuff[ri][rj]);
		      }  //if(rank == 25)printf("\n");
		   }

		   // free memory
		   free(recvBuff[0]);
		   free(recvBuff);
		}
	     }// end of "for(i...)"
	  }// end of "for(j...)"
       }// end of "if(isParent == 0 ||...)
}//recoveryByResampling()


//////////////////////////////////////////////////////////////////////////////////////////////////
void duplicateToSpareGrids(HaloArray2D *u, ProcGrid2D *g, MPI_Comm myCommWorld, GridCombine2D *gc, 
    int rank, int nprocs, int level){
    // Duplicating the values of upper grids to spare grids

    int i, j, fromGridId = 0, myGridId, toGridId = 0;
    MPI_Status nonFailedStatus;

    myGridId = gc->getGid(rank);

    // not spawned
    // considering: no more than one process failure
    if(myGridId >= 0 && myGridId < level){
       if(myGridId == 0){
	   toGridId = (3*level - 3);
       }
       else if(myGridId == (level-1)){
	   toGridId = (3*level - 2);
       }
       else if(myGridId > 0 && myGridId < (level-1)){
	   toGridId = (myGridId + 2*(level-1));
       }
       Vec2D<int> toGrid = gc->gridIx(toGridId);

       int c = g->P.x, d = g->P.y, k = g->id.x, l = g->id.y;
       int a = c, b = d;

       int startX, endX, startY, endY;
       int sk, si, indexX = 0, indexY = 0, destination, nLx, nLy;
       double lastIndexX, lastIndexY, nLastIndexX = 0.0, nLastIndexY = 0.0;
       double ** buff = NULL;
       int yIncrFlag = 0;

       if(!(isRightBoundary(g->id, g->P) || isUpperBoundary(g->id, g->P))){
	  nLx = u->l.x; nLy = u->l.y;
       }
       if(isRightBoundary(g->id, g->P) && !isUpperBoundary(g->id, g->P)){
	  nLx = u->l.x-1; nLy = u->l.y;
       }
       if(!isRightBoundary(g->id, g->P) && isUpperBoundary(g->id, g->P)){
	  nLx = u->l.x; nLy = u->l.y-1;
       }
       if(isRightBoundary(g->id, g->P) && isUpperBoundary(g->id, g->P)){
	  nLx = u->l.x-1; nLy = u->l.y-1;
       }

       lastIndexX = twoPow(toGrid.x)/(double)a;
       lastIndexY = twoPow(toGrid.y)/b;

       startY = 1; endY = (startY+nLy);
       for(j = 0; j < g->P.y; j++){ // j is fidy
	  startX = 1; endX = (startX+nLx);
	  for(i = 0; i < g->P.x; i++){ // i is fidx
	     // select group of process' grid coordinate who take part sending their portions to the mapped process on spare grids
	     if(((i < std::min(a, c)) && (k == i) && (j < std::min(b, d)) &&
		     (l == j)) || ((i >= std::min(a, c)) && (k == (i/2)) &&
		     (j < std::min(b, d)) && (l == j)) || ((i < std::min(a, c)) &&
		     (k == i) && (j >= std::min(b, d)) &&
		     (l == (j/2))) || ((i >= std::min(a, c)) &&
		     (k == (i/2)) && (j >= std::min(b, d)) && (l == (j/2)))){

		sk = startY; si = startX;
		yIncrFlag = 1;

		if(i != (a-1) && j != (b-1)){// allocate and initialize each with 0.0
		   nLastIndexX = lastIndexX;
		   nLastIndexY = lastIndexY;
		   buff = alloc2dDouble((int)nLastIndexX+2, (int)nLastIndexY+2);
		}
		if(i == (a-1) && j != (b-1)){
		   nLastIndexX = lastIndexX + 1;
		   nLastIndexY = lastIndexY;
		   buff = alloc2dDouble((int)nLastIndexX+2, (int)nLastIndexY+2);
		}
		if(i != (a-1) && j == (b-1)){
		   nLastIndexX = lastIndexX;
		   nLastIndexY = lastIndexY + 1;
		   buff = alloc2dDouble((int)nLastIndexX+2, (int)nLastIndexY+2);
		}
		if(i == (a-1) && j == (b-1)){
		   nLastIndexX = lastIndexX + 1;
		   nLastIndexY = lastIndexY + 1;
		   buff = alloc2dDouble((int)nLastIndexX+2, (int)nLastIndexY+2);
		}

		for(indexY = 0; indexY < (int)nLastIndexY; indexY++){
		  for(indexX = 0; indexX < (int)nLastIndexX; indexX++){
		      buff[indexX+1][indexY+1] = V(u, si+indexX, sk+indexY);
		     //printf("%d(%d,%d)%0.2f ", rank, indexX+1, indexY+1, buff[indexX+1][indexY+1]);
		  }
		  //printf("\n");
		}

		destination = getToRank(minRank(toGridId, nprocs, gc), i, j, a);
		//printf("[%d] Send to %d\n", rank, destination);

		MPI_Send(&(buff[0][0]), (int)(nLastIndexX+2)*(int)(nLastIndexY+2), MPI_DOUBLE, destination, 
			 NON_FAILED_FAILED_TAG, myCommWorld);
		// free the allocated memory for buff
		free(buff[0]);
		free(buff);

		//printf("[%d]: send values (%d, %d) ---> (%d, %d) to rank %d\n", rank, startX, startY, endX, endY, destination);
		startX = endX;
		endX = (startX + u->s.x - 2);
	     }// end of "if(((((k < ..."
	  }// end of "for(i...)"

	  if(yIncrFlag == 1){
	     startY = endY;
	     endY = (startY + u->s.y - 2);

	     yIncrFlag = 0;
	  }
       }// end of "for(j...)"
    }// end of if
    else if(myGridId >= (2*level - 1) && myGridId < (3*level - 1)){
       int c = g->P.x, d = g->P.y, k = g->id.x, l = g->id.y;
       int a, b, ri, rj, source;

       if(myGridId == (3*level - 3)){
	   fromGridId = 0;
       }
       else if(myGridId == (3*level - 2)){
	   fromGridId = (level-1);
       }
       else if(myGridId >=  (2*level -1) && myGridId < (3*level-1)){
	   fromGridId = (myGridId - 2*(level-1));
       }

       a = c; b = d;

       for(j = 0; j < b; j++){
	  for(i = 0; i < a; i++){
	     //if(((((k>i)?k-i:i-k) < right) || ((((k>i)?k-i:i-k) == 0))) && ((((l>j)?l-j:j-l) < upper) || (((l>j)?l-j:j-l) == 0))){
	     if(((i < std::min(a, c)) && (k == i) && (j < std::min(b, d)) &&
		    (l == j)) || ((i >= std::min(a, c)) && (k == (i/2)) &&
		    (j < std::min(b, d)) && (l == j)) || ((i < std::min(a, c)) &&
		    (k == i) && (j >= std::min(b, d)) &&
		    (l == (j/2))) || ((i >= std::min(a, c)) &&
		    (k == (i/2)) && (j >= std::min(b, d)) && (l == (j/2)))){

		source = getToRank(minRank(fromGridId, nprocs, gc), i, j, a);
		//printf("[%d] Receive from %d\n", rank, source);

		double ** recvBuff = alloc2dDouble(u->s.x, u->s.y);

		MPI_Recv(&(recvBuff[0][0]), (u->s.x)*(u->s.y), MPI_DOUBLE, source, NON_FAILED_FAILED_TAG, 
			 myCommWorld, &nonFailedStatus);

		for(rj = 0; rj < u->s.y; rj++){
		   for(ri = 0; ri < u->s.x; ri++){
		      *(u->ix(ri, rj)) = recvBuff[ri][rj];
		      //if(rank == 25)printf("%d(%d,%d)%0.2f ", source, ri, rj,  recvBuff[ri][rj]);
		   }  //if(rank == 25)printf("\n");
		}

		// free memory
		free(recvBuff[0]);
		free(recvBuff);

		//printf("[%d]: recv values from rank %d\n", rank, fromWho);
	     }
	  }// end of "for(i...)"
       }// end of "for(j...)"
    }// end of "else if...)
}//duplicateToSpareGrids()


/////////////////////////////////////////////////////////////////////////////////
void checkpointWrite2D(HaloArray2D *u, int rank){
    char fileName[128];
    int /*i, j, */MAX_PATH_LENGTH = 80, rowSize = u->s.x, colSize = u->s.y;
    char path[MAX_PATH_LENGTH];
    FILE *filePointer;

    //open file for write
    char * rv = getcwd(path, MAX_PATH_LENGTH);
    if(rv){
       sprintf(fileName, "%s/checkpoint_of_process_%d", path, rank);
    }
    else{
       printf("getcwd() failed\n\n");
       exit(1);
    }

    if(remove(fileName) == 0 ){
       if(rank == 0){
          printf("[%d] ===== %s (checkpoint file) deleted successfully =====\n\n", rank, fileName);
       }
    }
    else{
       printf("Unable to delete checkpoint file \n\n");
       //exit(1);
    }

    if((filePointer = fopen(fileName, "w")) == NULL){
       printf("fopen() for checkpoint write Error!!!\n\n");
       exit(1);
    }

    /*
    for(i = 0; i < rowSize; i++){
       if((int)(colSize) != (int)fwrite(writeBuff[i], sizeof(double), colSize, filePointer)){
          printf("fwrite() of checkpoint failed\n\n");
          exit(1);
       }
    }
    */

    /*
    if(rank == 4){
       //print contents of write buffer
       printf("Contents of writeBuff (rank %d) going to write is: \n", rank);
       for (j = 0; j < u->s.y; j++){
          for (i = 0; i < u->s.x; i++){
             printf("%0.2f ", *(u->ix(i, j)));
          }
          printf("\n");
       }
       printf("\n\n");
    }
    */

    if((int)(rowSize*colSize) != (int)fwrite(u->ix(0, 0), sizeof(double), rowSize*colSize, filePointer)){
       printf("fwrite() of checkpoint failed\n\n");
       exit(1);
    }

    //close file opened for writing
    fclose(filePointer);
}//checkpointWrite2D()

//////////////////////////////////////////////////////////////////////////////////
void checkpointRead2D(HaloArray2D *u, int rank){
    char fileName[128];
    int /*i, */MAX_PATH_LENGTH = 80, rowSize = u->s.x, colSize = u->s.y;
    char path[MAX_PATH_LENGTH];
    FILE *filePointer;

    //open file for reading
    char * rv = getcwd(path, MAX_PATH_LENGTH);
    if(rv){
       sprintf(fileName, "%s/checkpoint_of_process_%d", path, rank);
    }
    else{
       printf("getcwd() failed\n\n");
	   exit(1);
    }

    if((filePointer = fopen(fileName, "r")) == NULL){
       printf("fopen() for checkpoint read Error!!!\n\n");
       exit(1);
    }
    /*
    for(i = 0; i < rowSize; i++){
       if((int)(colSize) != (int)fread(readBuff[i], sizeof(double), colSize, filePointer)){
          printf("fread() of checkpoint failed\n\n");
          exit(1);
       }
    }
    */

    if((int)(rowSize*colSize) != (int)fread(u->ix(0, 0), sizeof(double), rowSize*colSize, filePointer)){
       printf("fread() of checkpoint failed\n\n");
       exit(1);
    }

    //close file opened for reading and delete them
    fclose(filePointer);
}//checkpointRead2D()


/////////////////////////////////////////////////////////////////////////////////////////
int getSlots(void) {
   FILE *f;
   char string[1000], seps[] = " \n,( )!=";
   char *p;
   f = fopen("hostfile","r");
   if(!f) {
      printf("Probably executing without a \"hostfile\" file. Returning slots = 1.\n");
      return 1; // return slots = 1
   }

   while(fgets(string, sizeof(string)-1, f) != NULL) {
      // Break into tokens
      p = string;
      // Find first token
      p = strtok(string, seps); 

      while(p != NULL){
         //printf("Token: %s\n", p);
         p = strtok(NULL, seps); // find next token

         if (strncmp("slots", p, strlen("slots")) == 0) {
            // "slots" token is found
            // our "desired token" (a number) is after "="
            p = strtok(NULL, seps); 
            fclose(f); // close the opened file
            return atoi(p); // convert string into int and return
         }
      }
   }
   fclose(f); // close the opened file

   // By default
   return 1;
} //getSlots


#endif /*FAILURERECOVERY_INCLUDED*/
