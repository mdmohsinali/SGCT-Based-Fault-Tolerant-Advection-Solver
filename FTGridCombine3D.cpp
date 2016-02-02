/* SGCT-Based-Fault-Tolerant-Advection-Solver Code source file.
   Copyright (c) 2015, Md Mohsin Ali. All rights reserved.
   Licensed under the terms of the BSD License as described in the LICENSE_FT_CODE file.
   This comment must be retained in any redistributions of this source file.
*/

// Parallel 2D Sparse Grid Combination class
// written by Peter Strazdins, May 2013
// updated by Mohsin Ali, September 2013, February 2014

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <assert.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "FTGridCombine3D.h"

#define GS_RECV_SEND_TAG 5 /*mesage tag for scatter recv and send comms*/
#define GS_SEND_RECV_TAG 6 /*mesage tag for gather send and recv comms*/

// return which grid id global rank is in
int GridCombine2D::getGid(int rank) {
  if (!(0 <= rank && rank < (nprocs+nprocsSg))) {
     printf("%d: getGid(bad rank=%d), nprocs=%d\n", myrank, rank, nprocs+nprocsSg);
  }
  //assert (0 <= rank && rank < nprocs); 
  assert (0 <= rank && rank < (nprocs+nprocsSg));
  
  int v;
  
  if(nprocsSg != 0 && rank >= nProcs(version, level, pD, fixedProcs)){
     v = nGrids();
  }
  else{
     v = gIds[rank];
  }
  
  if (fixedProcs) {
       if (rank < level * pD[0]) {
          if (v != rank / pD[0]){ 
             printf("%d: gId mismatch on rank %d: %d : %d\n", 
                  myrank, rank, v, rank / pD[0]);
         }
         assert (v == rank / pD[0]);
       } 
       else if (rank >= level * pD[0] && rank < (level * pD[0] + (level-1) * pD[1])) {              
         assert (v == level + (rank - level * pD[0]) / pD[1]);
       }            
       else if(nprocsSg != 0 && rank >= nProcs(version, level, pD, fixedProcs)){        
          assert (v == nGrids());
       }      
       
      if(version == CLASSIC_COMB  || version == CHKPOINT_RESTART){
         //do nothing      
      }
      else if(version == RESAMP_AND_COPY){    
         if(rank >= (level * pD[0] + (level-1) * pD[1]) && rank < nProcs(version, 
            level, pD, fixedProcs)){
            assert (v == nGridsTwoDiags() + (rank - (level*pD[0] + (level-1)*pD[1])) / pD[0]);
         }             
      }
      else if(version == ALTERNATE_COMB){       
        if(rank >= (level * pD[0] + (level-1) * pD[1]) && rank < (level * pD[0] + 
           (level-1) * pD[1] + (level-2) * (pD[1]/2)))   {
           assert (v == 2*level -1 + (rank - (level*pD[0] + (level-1)*pD[1])) / (pD[1]/2));
        }      
        else if(rank >= (level * pD[0] + (level-1) * pD[1] + (level-2) * (pD[1]/2)) && 
           rank < nProcs(version, level, pD, fixedProcs)){
           assert (v == 3*level - 3 + (rank - (level * pD[0] + (level-1) * pD[1] + 
              (level-2) * (pD[1]/2))) / (pD[1]/4));
        }       
      } 
  }
  return (v);
}

int GridCombine2D::nGrids() {
      int v = 0;
      
      if(version == CLASSIC_COMB  || version == CHKPOINT_RESTART){
         v = (2*level - 1); 
      }
      else if(version == RESAMP_AND_COPY){
         v = (3*level - 1); 
      }
      else if(version == ALTERNATE_COMB){    
         v = (4*level - 6);
      }
      
      return (v);
}

int GridCombine2D::nGridsTwoDiags() {
  return (2*level - 1);
}

int GridCombine2D::nGridsThreeDiags() {
  return (3*level - 3);
}

int GridCombine2D::diagRank(int gid) {
      int val = 0;
      
      if(gid < level)
         val = 0;
      else if(gid >= level && gid < nGridsTwoDiags())
         val = 1;      
  
      if(version == CLASSIC_COMB  || version == CHKPOINT_RESTART){
         //do nothing
      }
      else if(version == RESAMP_AND_COPY){
          if(gid >= nGridsTwoDiags() && gid < (nGrids()-2))
             val = 2;
          else if(gid >= (nGrids()-2) && gid < (nGrids()-1)) // grid on upper of grid 0 
             val = 3;  
          else if(gid >= (nGrids()-1) && gid < nGrids()) // grid on right of grid (level-1)
             val = 4;   
          else if(gid >= nGrids())
             val = 5;          
      }
      else if(version == ALTERNATE_COMB){  
         if(gid >= nGridsTwoDiags() && gid < nGridsThreeDiags())
            val = 2;
         else if(gid >= nGridsThreeDiags() && gid < nGrids())
            val = 3;   
         else if(gid >= nGrids())
            val = 4;
      }
      return (val);
}

int GridCombine2D::diagIx(int gid) {  
      int val = 0;
      
      if(gid < level)
         val = gid;
      else if(gid >= level && gid < nGridsTwoDiags())
         val = gid - level;      
  
      if(version == CLASSIC_COMB  || version == CHKPOINT_RESTART){
          //do nothing
      }
      else if(version == RESAMP_AND_COPY){
          if(gid >= nGridsTwoDiags() && gid < (nGrids()-2))
             val = gid - nGridsTwoDiags() + 1; // make them similar to grid 1 to grid (level-2)
          else if(gid >= (nGrids()-2) && gid < (nGrids()-1)) // grid on upper of grid 0 
             val = 0; // make it similar to grid 0 
          else if(gid >= (nGrids()-1) && gid < nGrids()) // grid on right of grid (level-1)
             val = (level-1); // make it similar to grid (level-1)           
      }
      else if(version == ALTERNATE_COMB){  
         if(gid >= nGridsTwoDiags() && gid < nGridsThreeDiags())
            val = gid - nGridsTwoDiags();
         else if(gid >= nGridsThreeDiags() && gid < nGrids())
            val = gid - nGridsThreeDiags();    
      }
      return (val);    
}

int GridCombine2D::getGrank(int rank) {
  //assert (0 <= rank && rank < nprocs); 
  assert (0 <= rank && rank < (nprocs+nprocsSg));
  int v = gRanks[rank];
  
  if (fixedProcs) {
       if (rank < level * pD[0]){
          assert (v == rank % pD[0]);
       }  
       else if (rank >= level * pD[0] && rank < (level * pD[0] + (level-1) * pD[1])) {  
          assert (v == (rank - level * pD[0])% pD[1]);          
       }  
       else if(nprocsSg != 0 && rank >= nProcs(version, level, pD, fixedProcs)){
          assert (v == (rank - nProcs(version, level, pD, fixedProcs)));
       } 
           
      if(version == CLASSIC_COMB  || version == CHKPOINT_RESTART){
         //do nothing       
      }
      else if(version == RESAMP_AND_COPY){   
          if(rank >= (level * pD[0] + (level-1) * pD[1]) && rank < nProcs(version, 
             level, pD, fixedProcs)){
             assert (v == (rank - (level*pD[0] + (level-1)*pD[1])) % pD[0]);
          }                 
      }
      else if(version == ALTERNATE_COMB){             
         if(rank >= (level * pD[0] + (level-1) * pD[1]) && rank < (level * pD[0] + 
            (level-1) * pD[1] + (level-2)*(pD[1]/2))){
            assert (v == (rank - (level*pD[0] + (level-1)*pD[1])) % (pD[1]/2));
         }   
         else if(rank >= (level * pD[0] + (level-1) * pD[1] + (level-2)*(pD[1]/2)) && 
            rank < nProcs(version, level, pD, fixedProcs)){
            assert (v == (rank - (level * pD[0] + (level-1) * pD[1] + 
               (level-2)*(pD[1]/2))) % (pD[1]/4));
         }
      }
  }
  return (v);
}

Vec2D<int> GridCombine2D::gridIx(int gid) {
  int id = gid;
  int adjustIndexX = gxS.x - (level-1);
  int adjustIndexY = gxS.y - (level-1);
  Vec2D<int> g;
  
  if(nprocsSg != 0 && gid == (nGrids())){
     assert (gid == nGrids());
     g.x = gxS.x,  g.y = gxS.y;
  }
  else{
      assert (0 <= gid  &&  gid < nGrids());      
      if (diagRank(gid) == 0) {
        g.x = gxS.x - gid,  g.y = gxS.y + gid - level + 1;
      } 
      else if (diagRank(gid) == 1) {
        gid -= level;
        g.x = gxS.x - gid - 1, g.y = gxS.y + gid - level + 1; 
      }       
      
      if(version == CLASSIC_COMB  || version == CHKPOINT_RESTART){
         //do nothing                
      }
      else if(version == RESAMP_AND_COPY){
             if(diagRank(gid) == 2){
               g.x = gxS.x - (gid - 2*(level-1)),  g.y = gxS.y + (gid - 2*(level-1)) - (level - 1); 
             }
             else if (diagRank(gid) == 3) { // grid on upper of grid 0 
               g.x = gxS.x,  g.y = gxS.y - (level - 1);
             }     
             else if (diagRank(gid) == 4) { // grid on right of grid (level-1)
               g.x = gxS.x - (level-1),  g.y = gxS.y;
             }            
      }
      else if(version == ALTERNATE_COMB){       
            if(diagRank(gid) == 2){
              gid -= 2*level - 1;
              g.x = gxS.x - gid - 2, g.y = gxS.y + gid - (level - 1); 
            }
            else if (diagRank(gid) == 3) {
              gid -= 3*level - 3;
              g.x = gxS.x - gid - 3, g.y = gxS.y + gid - (level - 1); 
            }        
            
            gridSizeToId[g.x - adjustIndexX][g.y - adjustIndexY] = id;
            gridIdToSize[id] = Vec2D<int>(g.x, g.y);            
      }
  }

  return g; 
}

GridCombine2D::GridCombine2D(int lv, int pd[2], bool fixedP, int sgProcs, 
			     Vec2D<int> gridS, Timer *t, int verb, bool dbg, 
                             bool isSeparate, int ver, MPI_Comm comm) {
  myCommWorld = comm;
  version = ver;
  gridSize = gridS;

  level = lv; pD[0] = pd[0]; pD[1] = pd[1]; fixedProcs = fixedP;
  timer = t;
  verbosity = verb; debug = dbg;
  MPI_Comm_rank(myCommWorld, &myrank);  // myrank also represents rank of separate sparse grid processes
  nprocs = nProcs(version, level, pD, fixedProcs);// only sub-grids in case of separate sparse grid case
  nprocsSg = isSeparate?(1<<sgProcs):0;           

  gRanks = new int[nprocs+nprocsSg]; gIds = new int[nprocs+nprocsSg];
  pgs = new ProcGrid2D* [isSeparate?(nGrids()+1):nGrids()];
  tempRank = isSeparate?((myrank>=nprocs)?-1:myrank):myrank;     

  if(version == CLASSIC_COMB  || version == CHKPOINT_RESTART){
     GridCombine2DClassicComb(sgProcs, gridS, isSeparate);          
  }
  else if(version == RESAMP_AND_COPY){
     GridCombine2DResampAndCopy(sgProcs, gridS, isSeparate);
  }
  else if(version == ALTERNATE_COMB){  
     GridCombine2DAlternateComb(sgProcs, gridS, isSeparate);
  }

   //printf("myrank = %d\n", myrank);
   pgU = pgs[getGid(myrank)]; // grid for this process   
   pgS = new ProcGrid2D((tempRank == -1 || (!isSeparate && myrank < (1<<sgProcs)))?myrank:-1, 
                    1 << sgProcs, myCommWorld, isSeparate?nprocs:0);  
  
  gxS = gridS;
  gxU = gridIx(getGid(myrank));

  selfBuf = 0;
  gatherSendBuf = 0;
  gatherSendBufSz = 0;
} //GridCombine2D()

void GridCombine2D::GridCombine2DClassicComb(int sgProcs, Vec2D<int> gridS, bool isSeparate) {
    
  int pCount = 0; // count of global processes covered so far
  int pInc = 0;
  int forLimit = isSeparate?(nGrids()+1):nGrids();
  
  for (int i = 0; i < forLimit; i++) { // define process grid for each grid
    if((isSeparate && tempRank != -1 && i < (forLimit-1)) || !isSeparate){
       pInc = nProcsPt(level-diagRank(i), pD[diagRank(i)], fixedProcs, 
			diagIx(i));
       for (int j = pCount; j < pCount + pInc; j++) { // setup process maps
         gRanks[j] = j - pCount;
         gIds[j] = i;
         // printf("%d: gIds[%d]=%d\n", rank, j, gIds[j]);
       }
       pgs[i] = new ProcGrid2D((pCount <= tempRank && tempRank < pCount+pInc)? tempRank: -1,
                           pInc, MPI_COMM_WORLD, pCount);
    }

    if(isSeparate && tempRank == -1 && i == (forLimit-1)){
       for (int k = 0; k < nprocsSg; k++) { // setup process maps
         gRanks[nprocs+k] = k;
         gIds[nprocs+k] = 2*level;
         // printf("%d: gIds[%d]=%d\n", rank, j, gIds[j]);
       }
    }
    pCount += pInc;
  }

  pCount = 0;
  if(isSeparate && tempRank == -1){
     for (int i = 0; i < nGrids(); i++) { // define process grid for each grid
        pInc = nProcsPt(level-diagRank(i), pD[diagRank(i)], fixedProcs, diagIx(i));
        for (int j = pCount; j < pCount + pInc; j++) { // setup process maps
          gRanks[j] = j - pCount;
          gIds[j] = i;
          // printf("%d: gIds[%d]=%d\n", rank, j, gIds[j]);
        }
        pCount += pInc;
     }
  }

  pCount = 0;
  if(isSeparate && tempRank == -1){
     for (int i = 0; i < nGrids(); i++) { // define process grid for each grid
        pInc = nProcsPt(level-diagRank(i), pD[diagRank(i)], fixedProcs, diagIx(i));

        for(int j = 0; j < nprocs; j++){
           if(getGid(j) == i){
              pgs[i] = new ProcGrid2D(j, pInc, MPI_COMM_WORLD, pCount);
           }
        }
        pCount += pInc;
     }
     //pgs[getGid(myrank)] = new ProcGrid2D(-1, 1 << sgProcs, MPI_COMM_WORLD, 0);
     pgs[getGid(myrank)] = new ProcGrid2D(myrank, 1 << sgProcs, MPI_COMM_WORLD, nprocs);
  }
}//GridCombine2DClassicComb()

void GridCombine2D::GridCombine2DResampAndCopy(int sgProcs, Vec2D<int> gridS, bool isSeparate) {
    
  int pCount = 0; // count of global processes covered so far
  int pInc = 0;
  int diagRankVal;
  int forLimit = isSeparate?(nGrids()+1):nGrids();
  
  for (int i = 0; i < forLimit; i++) { // define process grid for each grid
    if((isSeparate && tempRank != -1 && i < (forLimit-1)) || !isSeparate){
       if(diagRank(i) == 2 || diagRank(i) == 3 || diagRank(i) == 4){
           diagRankVal = 0;
       } 
       else{
           diagRankVal = diagRank(i);
       }
       pInc = nProcsPt(level-diagRankVal, pD[diagRankVal], fixedProcs, 
			diagIx(i));
       for (int j = pCount; j < pCount + pInc; j++) { // setup process maps
         gRanks[j] = j - pCount;
         gIds[j] = i;
       }
       pgs[i] = new ProcGrid2D((pCount <= tempRank && tempRank < pCount+pInc)? tempRank: -1,
                           pInc, myCommWorld, pCount);
    }

    if(isSeparate && tempRank == -1 && i == (forLimit-1)){
       for (int k = 0; k < nprocsSg; k++) { // setup process maps
         gRanks[nprocs+k] = k;
         gIds[nprocs+k] = nGrids();
       }
    }
    pCount += pInc;
  }

  pCount = 0;
  if(isSeparate && tempRank == -1){
     for (int i = 0; i < nGrids(); i++) { // define process grid for each grid
       if(diagRank(i) == 2 || diagRank(i) == 3 || diagRank(i) == 4){
           diagRankVal = 0;
       } 
       else{
           diagRankVal = diagRank(i);
       }       
       pInc = nProcsPt(level-diagRankVal, pD[diagRankVal], fixedProcs, 
			diagIx(i));
       for (int j = pCount; j < pCount + pInc; j++) { // setup process maps
         gRanks[j] = j - pCount;
         gIds[j] = i;
       }         
        pCount += pInc;
     }
  }
  
  pCount = 0;
  if(isSeparate && tempRank == -1){
     for (int i = 0; i < nGrids(); i++) { // define process grid for each grid
       if(diagRank(i) == 2 || diagRank(i) == 3 || diagRank(i) == 4){
            diagRankVal = 0;
        }          
       else{
           diagRankVal = diagRank(i);
       }        
        pInc = nProcsPt(level-diagRankVal, pD[diagRankVal], fixedProcs, diagIx(i));

        for(int j = 0; j < nprocs; j++){
           if(getGid(j) == i){
              pgs[i] = new ProcGrid2D(j, pInc, myCommWorld, pCount);
           }
        }
        pCount += pInc;
     }
     //pgs[getGid(myrank)] = new ProcGrid2D(-1, 1 << sgProcs, MPI_COMM_WORLD, 0);
     pgs[getGid(myrank)] = new ProcGrid2D(myrank, 1 << sgProcs, myCommWorld, nprocs);
  }  
}//GridCombine2DResampAndCopy()


void GridCombine2D::GridCombine2DAlternateComb(int sgProcs, Vec2D<int> gridS, bool isSeparate) {
    
   int pCount = 0; // count of global processes covered so far
   int pInc = 0;
   int diagRankVal;
   int pDval;
   int forLimit = isSeparate?(nGrids()+1):nGrids();    

   gridSizeToId = new int* [level];
   for(int i = 0; i < level; i++){
      gridSizeToId[i] = new int [level];
   }

   #pragma omp parallel for default(shared)
   for(int i = 0; i < level; i++){
      for(int j = 0; j < level; j++){
         gridSizeToId[i][j] = -1;
      }
   }

   gridIdToSize = new Vec2D<int> [nGrids()];
   
   #pragma omp parallel for default(shared)
   for(int i = 0; i < nGrids(); i++){
      gridIdToSize[i] = Vec2D<int>(-1, -1);
   }

   for (int i = 0; i < forLimit; i++) { // define process grid for each grid
     if((isSeparate && tempRank != -1 && i < (forLimit-1)) || !isSeparate){
        diagRankVal = diagRank(i);
        if(diagRank(i) < 2){
           pDval = pD[diagRankVal];
        } 
        else{
           pDval = pD[1]/(pow(2, diagRank(i)-1));
        }
        pInc = nProcsPt(level-diagRankVal, pDval, fixedProcs, diagIx(i));
        for (int j = pCount; j < pCount + pInc; j++) { // setup process maps
           gRanks[j] = j - pCount;
           gIds[j] = i;
        }
        pgs[i] = new ProcGrid2D((pCount <= tempRank && tempRank < pCount+pInc)? tempRank: -1,
                        pInc, myCommWorld, pCount);
     }

     if(isSeparate && tempRank == -1 && i == (forLimit-1)){
        for (int k = 0; k < nprocsSg; k++) { // setup process maps
           gRanks[nprocs+k] = k;
           gIds[nprocs+k] = nGrids();
        }
     }
     pCount += pInc;
   }

   pCount = 0;
   if(isSeparate && tempRank == -1){
     for (int i = 0; i < nGrids(); i++) { // define process grid for each grid
        diagRankVal = diagRank(i);
        if(diagRank(i) < 2){
           pDval = pD[diagRankVal];
        } 
        else{
           pDval = pD[1]/(pow(2, diagRank(i)-1));
          }    
        pInc = nProcsPt(level-diagRankVal, pDval, fixedProcs, diagIx(i));
        for (int j = pCount; j < pCount + pInc; j++) { // setup process maps
           gRanks[j] = j - pCount;
           gIds[j] = i;
        }         
        pCount += pInc;
      }
   }
  
   pCount = 0;
   if(isSeparate && tempRank == -1){
      for (int i = 0; i < nGrids(); i++) { // define process grid for each grid
         diagRankVal = diagRank(i);
         if(diagRank(i) < 2){
            pDval = pD[diagRankVal];
         } 
         else{
            pDval = pD[1]/(pow(2, diagRank(i)-1));
         }      
         pInc = nProcsPt(level-diagRankVal, pDval, fixedProcs, diagIx(i));
         for(int j = 0; j < nprocs; j++){
            if(getGid(j) == i){
               pgs[i] = new ProcGrid2D(j, pInc, myCommWorld, pCount);
            }
         }
         pCount += pInc;
      }
      //pgs[getGid(myrank)] = new ProcGrid2D(-1, 1 << sgProcs, MPI_COMM_WORLD, 0);
      pgs[getGid(myrank)] = new ProcGrid2D(myrank, 1 << sgProcs, myCommWorld, nprocs);
   }
}//GridCombine2DAlternateComb()

GridCombine2D::~GridCombine2D() {
  for (int i=0; i < nGrids(); i++){
      delete pgs[i];
  }    
  delete[] pgs; 
  
  if(version == ALTERNATE_COMB){
      for (int i = 0; i < level; i++){
          delete[] gridSizeToId[i];  
      }
      delete[] gridSizeToId;
     
      delete[] gridIdToSize;
  }
  
  delete pgS; delete[] gRanks; delete[] gIds;  
}
/* 
Implementation notes: current implementation uses buffered send. This
requires providing MPI_Buffer_attach() a large buffer. This cannot be
released until all processes reach a barrier.
To use ISend/IRecv(), we would need to:
   1. create a list of handles and buffers covering all outgoing messages
      For gatherSend(), the number of messages is bounded by s.x*s.y where 
         s = pgS->P / ppU->P + 2
   2. creates a list of handles for all incoming messages.        
      For gatherRecv(), number received is bounded by Sum i: s[i].x*s[i].y 
         s[i] = pgU[i]->P / pgS->P + 2
      We would also need to have lists recording uR, ij & dn.
   3. do MPI_WaitAll() on outgoing handles (now buffers can be released) 
   4. do MPI_WaitAll() on incoming handles and complete interpolation
-----
Done
 */

//void GridCombine2D::gatherScatter(HaloArray2D* u, HaloArray2D* usg) {
void GridCombine2D::gatherScatter(HaloArray2D* u, HaloArray2D* usg, 
     int * failedList, int numFailed, MPI_Comm myComm) {
  if(version == CLASSIC_COMB  || version == CHKPOINT_RESTART){
     gatherScatterAllVersions(u, usg, failedList, numFailed, myComm);          
  }
  else if(version == RESAMP_AND_COPY && ((getGid(pgU->myrank) < nGridsTwoDiags()) || 
     (getGid(pgU->myrank) >= nGrids()))){
     gatherScatterAllVersions(u, usg, failedList, numFailed, myComm);
  }
  else if(version == ALTERNATE_COMB){
     gatherScatterAllVersions(u, usg, failedList, numFailed, myComm);      
  }

} //gatherScatter()


//gatherScatterAllVersions()
void GridCombine2D::gatherScatterAllVersions(HaloArray2D* u, HaloArray2D* usg, 
     int * failedList, int numFailed, MPI_Comm myComm) {
  timer->start("gridCombine", 0, 0);
  if(tempRank != -1){ 
     gatherSendScatterRecvInit();
  }
    
  if(tempRank == -1 || nprocsSg == 0){
     gatherRecvScatterSendInit();
  }
  
  if(tempRank != -1){ 
     gatherSend(u);       
  }

  if(tempRank == -1 || nprocsSg == 0){
     gatherRecv(usg, failedList, numFailed);       
  }

  if(tempRank != -1){  
     MPI_Waitall(gatherSendCounter, nGatherSendRequests, nGatherSendStatuses);
     for(int i = 0; i < gatherSendCounter; i++){
        delete gatherSendBuff[i];
     }
     delete[] gatherSendBuff;
  }
   
  if(tempRank == -1 || nprocsSg == 0){
     if (pgS->myrank >= 0){
        for(int i = 0; i < gatherRecvCounter; i++){                  
            delete guR[i];                                           
        }
        delete[] guR;        
     } 
  }

  if(tempRank == -1 || nprocsSg == 0){
     scatterSend(usg, failedList, numFailed);
  }

  if(tempRank != -1){
     scatterRecv(u);
  }

  if(tempRank == -1 || nprocsSg == 0){
     if (pgS->myrank >= 0){
        MPI_Waitall(scatterSendCounter, nScatterSendRequests, nScatterSendStatuses); 
        for(int i = 0; i < scatterSendCounter; i++){
            delete scatterSendBuff[i];
        }
        delete[] scatterSendBuff; delete[] gatherRecvCoeff;
        delete[] grSU; delete[] gdn; delete[] gi; delete[] gj;
     }
     
  delete[] nScatterSendRequests;
  delete[] nScatterSendStatuses;

  delete[] nGatherRecvRequests;
  delete[] nGatherRecvStatuses;  
  }
  
  if(tempRank != -1){
     for(int i = 0; i < scatterRecvCounter; i++){
         delete suR[i];
     }
     delete[] suR; 
     delete[] sdn; delete[] si; delete[] sj;    

     delete[] nGatherSendRequests; 
     delete[] nGatherSendStatuses;

     delete[] nScatterRecvRequests;
     delete[] nScatterRecvStatuses;
  }
   
  timer->stop("gridCombine");
} //gatherScatterAllVersions()


//gatherSendScatterRecvInit()
void GridCombine2D::gatherSendScatterRecvInit(){
  Vec2D<int> numMsgs = pgS->P / pgU->P;
  nGatherSends = (numMsgs.x)*(numMsgs.y);
  nScatterRecvs = (numMsgs.x)*(numMsgs.y);  

  nGatherSendRequests = new MPI_Request[nGatherSends];  
  nGatherSendStatuses = new MPI_Status[nGatherSends]; 

  gatherSendBuff = new double*[nGatherSends];    

  nScatterRecvRequests = new MPI_Request[nScatterRecvs];
  nScatterRecvStatuses = new MPI_Status[nScatterRecvs];  
  sdn = new Vec2D<int>[nScatterRecvs];  
  si = new int[nScatterRecvs];
  sj = new int[nScatterRecvs];    
  suR = new HaloArray2D *[nScatterRecvs];
}
//gatherSendScatterRecvInit()


//gatherRecvScatterSendInit()
void GridCombine2D::gatherRecvScatterSendInit(){
  if (pgS->myrank >= 0){ // this process holds part of sparse grid
      if(version == CLASSIC_COMB  || version == CHKPOINT_RESTART){
         nScatterSends = nGrids();
         nGatherRecvs = nGrids();                    
      }
      else if(version == RESAMP_AND_COPY){
         nScatterSends = nGridsTwoDiags();
         nGatherRecvs = nGridsTwoDiags();          
      }
      else if(version == ALTERNATE_COMB){
         nScatterSends = nGrids();
         nGatherRecvs = nGrids();
      }      
  }
  else{ // this process does not hold part of sparse grid
      nScatterSends = 0;
      nGatherRecvs = 0; 
  }    
  //printf("                            [%d] nGatherSends = %d\n", pgU->myrank, nGatherSends);     
  
  nGatherRecvRequests = new MPI_Request[nGatherRecvs];  
  nGatherRecvStatuses = new MPI_Status[nGatherRecvs];   

  grSU = new Vec2D<int>[nGatherRecvs];                  
  gdn = new Vec2D<int>[nGatherRecvs];                     
  gi = new int[nGatherRecvs];                           
  gj = new int[nGatherRecvs];                           

  guR = new HaloArray2D *[nGatherRecvs];                
  gatherRecvCoeff = new double[nGatherRecvs];           
  
  nScatterSendRequests = new MPI_Request[nScatterSends];
  nScatterSendStatuses = new MPI_Status[nScatterSends];  
  scatterSendBuff = new double*[nScatterSends];
}
//gatherRecvScatterSendInit()

// current process sends its grid to the respective sparse grid process.
// There may be > 1 of these when more processes are used for
// the sparse grid than the current grid (the expected case). 
// nb. grid dimensions are always <= those of the sparse grid (=> / rSU>={1,1})
void GridCombine2D::gatherSend(HaloArray2D* u) {
  timer->start("gatherSend", u->l.prod(), 1);
  gatherSendScatterRecv(true, u);
  timer->stop("gatherSend");
}
void GridCombine2D::scatterRecv(HaloArray2D* u) {
  timer->start("scatterRecv", u->l.prod(), 1);
  gatherSendScatterRecv(false, u);
  timer->stop("scatterRecv");
}
   
void GridCombine2D::gatherSendScatterRecv(bool send, HaloArray2D* u) {
  Vec2D<int> rSU = gridSz1(gxS) / gridSz1(gxU); 
  Vec2D<int> NU = gridSz(gxU), NS = gridSz(gxS);

  // now find SG co-ordinates and offsets corresponding to our first point
  Vec2D<int> srcG0 = pgU->L2G0(NU), destG0 = srcG0 * rSU; 

  Vec2D<int> p0S = pgS->getP0(destG0, NS);  
  Vec2D<int> offs0S = pgS->getOffs0(destG0, NS);
  Vec2D<int> nU = pgU->G2L(NU);      // we will send away all of our points
  int i=0; Vec2D<int> pS = p0S, offsS = offs0S;
  gatherSendCounter = 0;
  scatterRecvCounter = 0;  

  while (i < nU.x) {
    int j=0, dnx = nU.y==0? nU.x: 0; // force termination if next loop is empty
    pS.y = p0S.y; offsS.y = offs0S.y;
    while (j < nU.y) {
      // following case not yet implemented; will require sending prev. points.
      // Should not occur if pgS process grid sizes are powers of 2

      assert (pgS->L2G0(NS, pS) % rSU == Vec2D<int>(0,0));
      assert (pS <= pgS->P); // check if not stopped when should or bad getP0()
      si[scatterRecvCounter] = i; sj[scatterRecvCounter] = j;    
      Vec2D<int> ij = Vec2D<int>(i, j);
      Vec2D<int> nS = pgS->G2L(NS, pS) - offsS; // num. points on dest
      Vec2D<int> dn = nS / rSU;         // corresp. number of points here
      assert (pgS->ownsData(pS, destG0 + rSU*ij, rSU*dn, NS)); // sanity check
      // now add last row/col (grid length=2^k+1), if truncated by / rSU above 
      // o.w. the case nS % rSU != (0,0) is covered as we pass next row/col
      dn = dn + (pgS->lastProc(pS) % rSU); // add if dest has them
      dn = dn.min(nU - ij);             // truncate to the end of our points
      sdn[scatterRecvCounter] = dn;   
      int rankS = pgS->getRank(pS.x, pS.y);

      if (verbosity > 1)
         printf("%d: %s %dx%d pts from (%d,%d) to/from %d=(%d,%d)\n", 
	       pgU->myrank, send? "gather send": "scatter recv", dn.x, dn.y, 
	       srcG0.x+i, srcG0.y+j, rankS, pS.x, pS.y);

      assert (dn.prod() > 0);  // as we still have points to send!
      // we need to include next row/col for interpolation as rSU > {1,1}
      if (send) {
	timer->start("pack", u->l.prod(), 2);
        gatherSendBuff[gatherSendCounter] = u->pack(i, j, dn.x+1, dn.y+1);
	timer->stop("pack");
        if (rankS != pgU->myrank) {
           MPI_Isend(gatherSendBuff[gatherSendCounter], (dn+1).prod(), MPI_DOUBLE, rankS, 
              GS_SEND_RECV_TAG, pgS->comm, &nGatherSendRequests[gatherSendCounter]);             
           gatherSendCounter++;
           //printf("         [%d] gatherSend: sending to %d\n", pgU->myrank, rankS); 
        } 
        else 
          selfBuf = gatherSendBuff[gatherSendCounter]; 
      }//end of send 

      else { // recv
	HaloArray2D *uR = new HaloArray2D(dn, 0);
        suR[scatterRecvCounter] = new HaloArray2D(dn, 0); 
	if (rankS != pgS->myrank) {
	   //MPI_Status s; int rv, count;
	   //MPI_Recv(uR->u, dn.prod(), MPI_DOUBLE, rankS, GS_RECV_SEND_TAG, pgS->comm, &s);
           MPI_Irecv(suR[scatterRecvCounter]->u, dn.prod(), MPI_DOUBLE, rankS, GS_RECV_SEND_TAG, 
              pgS->comm,  &nScatterRecvRequests[scatterRecvCounter]);            
           //printf("                    [%d] ScatterRecv: receiving from %d\n", pgU->myrank, rankS);
           scatterRecvCounter++;         
	   //rv = MPI_Get_count(&s, MPI_DOUBLE, &count);
	   //assert(rv != MPI_ERR_TRUNCATE);
	   //assert(count == dn.prod()); // we got the size we expected
        }
        else {            
	  delete[] (uR->u);
          assert (selfBuf!=0); // set by scatterSend()
	  uR->u = selfBuf;            
	  selfBuf = 0;
	}       
	timer->start("unpack", dn.prod(), 2);
        u->unpack(uR->u, i, j, dn.x, dn.y);
	timer->stop("unpack");   
        delete uR;
      }
      j += dn.y; dnx = dn.x;
      pS.y++; offsS.y = 0;
    
    } //for (j...)
    i += dnx;
    pS.x++; offsS.x = 0;
  } //for (i...)

  // Unpack asynchronously received (scatterReceive) values
  int waitAnyCounter, waitAnyIndex;

  if (!send) { //recv
     for(waitAnyCounter = 0; waitAnyCounter < scatterRecvCounter; waitAnyCounter++){
        MPI_Waitany(scatterRecvCounter, nScatterRecvRequests, &waitAnyIndex, 
           nScatterRecvStatuses);
        //printf("                rank = %d, scatterRecvCounter = %d, waitAnyIndex = %d\n", pgS->myrank, scatterRecvCounter, waitAnyIndex);
	timer->start("unpack", sdn[waitAnyIndex].prod(), 2);
        u->unpack(suR[waitAnyIndex]->u, si[waitAnyIndex], sj[waitAnyIndex], 
           sdn[waitAnyIndex].x, sdn[waitAnyIndex].y);
	timer->stop("unpack");   
     } // end of for(waitAnyCounter...
  }
} //gatherSend()

// for each component grid, gather contributions from respective processes. 
// There may be > 1 of these when fewer processes are used for
// the sparse grid than the component grid. 
void GridCombine2D::gatherRecv(HaloArray2D* uS, int* failedList, int numFailed) {
  timer->start("gatherRecv", uS->l.prod(), 1);
  timer->start("zero", uS->l.prod(), 2);
  uS->zero(); 
  timer->stop("zero");
  gatherRecvScatterSend(true, uS, failedList, numFailed);
  timer->stop("gatherRecv");
}

void GridCombine2D::scatterSend(HaloArray2D* uS, int* failedList, int numFailed) {
  timer->start("scatterSend", uS->l.prod(), 1);
  gatherRecvScatterSend(false, uS, failedList, numFailed);
  timer->stop("scatterSend");
}

void GridCombine2D::gatherRecvScatterSend(bool recv, HaloArray2D* uS, int* failedList, int numFailed) {      
  if (pgS->myrank < 0) // this process does not hold part of sparse grid
    return;

  scatterSendCounter = 0; 
  gatherRecvCounter = 0;   

  double * listCoeffs = NULL;

  if(version == ALTERNATE_COMB){
     double sTime = 0.0, eTime;    
     if(myrank == 0){     
        sTime = MPI_Wtime();
     }
     if(recv){
        listCoeffs = new double[nGrids()]; //should be released
        selectCombCoeffs(failedList, numFailed, listCoeffs);
            
        if(myrank == 0 && verbosity > 0){
           printf("\n");  
           for(int i = 0; i < nGrids(); i++){                                             
              printf("[%d] ===== Coefficient of grid %d = %0.1f =====\n", myrank, i, listCoeffs[i]);
           }
           printf("\n"); 
        }
        if(myrank == 0){
           eTime = MPI_Wtime();         
           printf("[%d]----- Creating coefficient list takes %0.6f Sec (MPI_Wtime) -----\n", 
                   myrank, eTime - sTime);
        }        
     }
  }

  for (int g = 0; g < ((version == RESAMP_AND_COPY)? nGridsTwoDiags(): nGrids()); g++) {
    //double coeff = (debug || diagRank(g)==0)? +1.0: -1.0;

    double coeff = 0.0;
    if(version == CLASSIC_COMB  || version == RESAMP_AND_COPY || version == CHKPOINT_RESTART){
       coeff = (debug || diagRank(g)==0)? +1.0: -1.0; 
    }
    else if(version == ALTERNATE_COMB){
       coeff = (debug)? +1: ((recv)? listCoeffs[g]: 0.0); 
    }

    gatherRecvCoeff[gatherRecvCounter] = coeff;  
    Vec2D<int> gxU = gridIx(g);
    ProcGrid2D * pgU = pgs[g];
    Vec2D<int> rSU = gridSz1(gxS) / gridSz1(gxU);
    grSU[gatherRecvCounter] = gridSz1(gxS) / gridSz1(gxU);  
    Vec2D<int> NU = gridSz(gxU), NS = gridSz(gxS);

    // grid U process co-ordinates and offsets corresponding to our first point
    Vec2D<int> destG0 = pgS->L2G0(NS), srcG0 = destG0 / rSU; 
    Vec2D<int> p0U = pgU->getP0(srcG0, NU);
    Vec2D<int> offs0U = pgU->getOffs0(srcG0, NU);  
    Vec2D<int> nS = pgS->G2L(NS);
    int i=0; Vec2D<int> pU = p0U, offsU = offs0U;

    while (i < nS.x) {
      int j=0, dnx = nS.y==0? nS.x: 0;
      pU.y = p0U.y; offsU.y = offs0U.y;
      while (j < nS.y) {
        assert (pU <= pgU->P); // check if not stopped when should or bad getP0
	Vec2D<int> ij = Vec2D<int>(i, j);
        gi[gatherRecvCounter] = i; gj[gatherRecvCounter] = j;    
        Vec2D<int> nU = pgU->G2L(NU, pU) - offsU; // number of points on sender
        Vec2D<int> dn = nU * rSU;  // corresponding number of points here
        dn = dn.min(nS - ij);      // cut off at our boundary 
        gdn[gatherRecvCounter] = dn;   
	nU = nU.min(dn / rSU);     // do not expect extra points to be sent
	Vec2D<int> lastPts = (pgU->lastProc(pU)).min(pgS->lastProc()) % rSU;
        nU = nU + lastPts;         // unless they're the last ones  
        int rankU = pgU->getRank(pU.x, pU.y);

        if (verbosity > 1)
	   printf("%d: %s %dx%d pts grid %d proc %d=%d,%d corresp. pt (%d,%d) %dx%d\n", 
		 pgS->myrank, recv? "gather recv": "scatter send", nU.x, nU.y, g, rankU, 
		 pU.x, pU.y, destG0.x+i, destG0.y+j, dn.x, dn.y);

        assert (dn.x > 0  &&  dn.y > 0); // as we have more points to receive
        assert (pgU->ownsData(pU, srcG0 + ij/rSU, nU, NU)); // sanity check

	if (recv) {
	  nU = nU + 1; // include next points for interpolation
	  HaloArray2D *uR = new HaloArray2D(nU, 0);
	  guR[gatherRecvCounter] = new HaloArray2D(nU, 0);  
	  if (rankU != pgS->myrank) {
	    //MPI_Status s; int rv, count;
	    //MPI_Recv(uR->u, nU.prod(), MPI_DOUBLE, rankU, GS_SEND_RECV_TAG, pgS->comm, &s);
	    MPI_Irecv(guR[gatherRecvCounter]->u, nU.prod(), MPI_DOUBLE, rankU, GS_SEND_RECV_TAG, 
                      pgS->comm, &nGatherRecvRequests[gatherRecvCounter]);            
            //printf("                    [%d] gatherRecv: receiving from %d\n", pgS->myrank, rankU);        
            gatherRecvCounter++;                          
            //printf("                rank = %d, rankU = %d, gatherRecvCounter = %d\n", pgS->myrank, rankU, gatherRecvCounter);            
	    //rv = MPI_Get_count(&s, MPI_DOUBLE, &count);
	    //assert(rv != MPI_ERR_TRUNCATE);
	    //assert(count == nU.prod()); // we got the size we expected
          } else {
	    delete[] (uR->u);
            assert (selfBuf!=0); // set by gatherSend()
	    uR->u = selfBuf;
            if (verbosity >= 4)
              uR->print(pgS->myrank, "uR");
            timer->start("interpolate", (uR->l * rSU).prod(), 2);
            uS->interpolate(coeff, uR, rSU, i, j, dn);
            timer->stop("interpolate");
	    selfBuf = 0;
	  }
          delete uR;
	} 
        else { // send
	  timer->start("sample", uS->l.prod(), 2);
          scatterSendBuff[scatterSendCounter] = uS->sample(i, j, rSU, nU);  
	  timer->stop("sample");
          if (rankU != pgS->myrank) {
	     MPI_Isend(scatterSendBuff[scatterSendCounter], nU.prod(), MPI_DOUBLE, rankU, 
                       GS_RECV_SEND_TAG, pgS->comm, &nScatterSendRequests[scatterSendCounter]);  
             //printf("         [%d] ScatterSend: sending to %d\n", pgS->myrank, rankU);                
             scatterSendCounter++;  
        } else 
            selfBuf = scatterSendBuff[scatterSendCounter]; 
	}

        j += dn.y; dnx = dn.x;
        pU.y++; offsU.y = 0;
      } //while (j...)

      i += dnx;
      pU.x++; offsU.x = 0;

    } //while (i...)

  } // while (g...)
  // Interpolate asynchronously received (gatherReceive) values
  int waitAnyCounter, waitAnyIndex;
     
  if (recv) {               
     for(waitAnyCounter = 0; waitAnyCounter < gatherRecvCounter; waitAnyCounter++){
        MPI_Waitany(gatherRecvCounter, nGatherRecvRequests, &waitAnyIndex, 
           nGatherRecvStatuses);
        //printf("                rank = %d, gatherRecvCounter = %d, waitAnyIndex = %d\n", pgS->myrank, gatherRecvCounter, waitAnyIndex);
        if (verbosity >= 4){
           guR[waitAnyIndex]->print(pgS->myrank, "guR[waitAnyInedx]");
        }    
        timer->start("interpolate", (guR[waitAnyIndex]->l * grSU[waitAnyIndex]).prod(), 
                      2);     
        uS->interpolate(gatherRecvCoeff[waitAnyIndex], guR[waitAnyIndex], grSU[waitAnyIndex], 
                        gi[waitAnyIndex], gj[waitAnyIndex], gdn[waitAnyIndex]);          
        timer->stop("interpolate");                  
     } // end of for(waitAnyCounter...
  }
  if(listCoeffs != NULL){//!NULL for version ALTERNATE_COMB and NULL for others
     delete[] listCoeffs;
  }
} //gatherRecvScatterSend()


void GridCombine2D::selectCombCoeffs(int * failedList, int numFailed, double * listCoeffs){
   int * isMax = new int[nGrids()];
   double * coeffList = new double[nGrids()];
   double * coeffListAlt = new double[nGrids()];
   int * isGridFailed = new int[nGrids()];
   int * maxGridList = new int [level]; 
   int fid, gid, levId, levId2, failedGridCounter = 0, dimSizeY, dimSizeX, totMax = 0, maxFound;   
   int adjustIndexX = gxS.x - (level-1);
   int adjustIndexY = gxS.y - (level-1); 
   
   int ** maxGridList2D = new int *[level];
   for(levId = 0; levId < level; levId++){
      maxGridList2D[levId] = new int [level-levId];
   }   

   // Initialization
   #pragma omp parallel for default(shared)
   for(gid = 0; gid < nGrids(); gid++){
      isMax[gid] = 0;
      coeffList[gid] = 0.0;
      coeffListAlt[gid] = 0.0;
      isGridFailed[gid] = 0;
   }

   #pragma omp parallel for default(shared)
   for(levId = 0 ; levId < level; levId++){
      maxGridList[levId] = -1; 
   }

   #pragma omp parallel for default(shared)
   for(levId = 0; levId < level; levId++){
      for(levId2 = 0; levId2 < (level-levId); levId2++){       
         maxGridList2D[levId][levId2] = -1;
      }
   }    

   // Determine which grids are failed
   #pragma omp parallel for default(shared)
   for(fid = 0; fid < numFailed; fid++){
      isGridFailed[getGid(failedList[fid])] = 1;
   }

   // Determine number of failed grids
   for(gid = 0; gid < nGrids(); gid++){
      if(isGridFailed[gid] == 1){
          failedGridCounter++; // I have to use it for testing the condition of how many grid is possible to fail
      }
   }
   
   // Check if all grids are failed. exit if so
   if(failedGridCounter == nGrids()){
       if(myrank == 0){
          printf("\n[%d] ***** All grids are failed. Exiting the application. *****\n\n", myrank);
       }
       exit(1);
   }

   // Determine gridIdToSize and gridSizeToId
   for(gid = 0; gid < nGrids(); gid++){
      gridIx(gid);
   }   
   
   // Determine a list which contains max
   for(gid = 0; gid < nGrids(); gid++){
      maxFound = 0;//reset 
      dimSizeY = gridIdToSize[gid].y;         
      dimSizeX = gridIdToSize[gid].x;               
      for(levId2 = (dimSizeX - adjustIndexX); levId2 < (level - (dimSizeY - adjustIndexY)); levId2++){
          for(levId = (dimSizeY - adjustIndexY); levId < (level - levId2); levId++){
              if(maxGridList2D[levId2][levId] != -1){
                 maxFound = 1;
                 break;
              }
          }
          if(maxFound == 1){
             break;          
          }
      }
      if(!isGridFailed[gid] && !maxFound){          
         maxGridList[dimSizeY - adjustIndexY] = gid;    
         maxGridList2D[dimSizeX - adjustIndexX][dimSizeY - adjustIndexY] = gid;        
         totMax++;
      }
   } 

   // Printing max grid list
   if(myrank == 0 && verbosity > 0){
      for(levId = 0 ; levId < level; levId++){
         printf("[%d] ***** maxGridList[%d] = %d *****\n", myrank, levId, maxGridList[levId]);
      }   
   }
   
   // Calculating coefficients
   // Scanning the max values from bottom-to-top and top-to-bottom order
   // Generate two sets of coefficient values
   // Choose the one which contains both the first and last grid of upper layer (if possible)
   int onlyContribGrid = 0, minusCoeffCounter = 0;
   if(totMax > 0){
      int nextLevId, failedTest, intersectId, xSize, ySize;  
      // Scanning from bottom-to-top
      for(levId = 0; levId < level; levId++){
         while(maxGridList[levId] == -1 && levId < level){//this does not hold max value, try for next grid
            levId++;
         }
         if(levId < level){
            onlyContribGrid = maxGridList[levId];
         }

         for(nextLevId = (levId+1); nextLevId < level; nextLevId++){
            while(maxGridList[nextLevId] == -1 && nextLevId < level){//this does not hold max value, try for next grid
               nextLevId++;
            }           
      
            if(nextLevId < level){ //at least two max values found
               xSize = gridIdToSize[maxGridList[nextLevId]].x;
               ySize = gridIdToSize[maxGridList[levId]].y;

               intersectId = gridSizeToId[xSize - adjustIndexX][ySize - adjustIndexY];
               failedTest = (intersectId == -1)? 1: isGridFailed[intersectId];
 
               if(failedTest == 1){ //intersection is failed or invalid
                  continue;
               }
               else if(failedTest == 0){ //intersection is not failed or not invalid
                  coeffList[maxGridList[levId]] = +1.0;
                  coeffList[maxGridList[nextLevId]] = +1.0;
                  coeffList[intersectId] = -1.0;
	
                  minusCoeffCounter++;	          

                  levId = nextLevId-1; // replace levId with nextLevId in outer loop (after increment)
                  nextLevId = level-1; //break; //breaking inner loop (condition will fail after the increment in loop)
               }
            }//end of if(nextLevId < level...
         }//end of for(nextLevId = ...
      }//end of for(levId = 0...

      // Scanning from top-to-bottom     
      for(levId = level-1; levId >= 0; levId--){
         while(maxGridList[levId] == -1 && levId >= 0){//this does not hold max value, try for next grid
            levId--;
         }
         if(levId >= 0){
            onlyContribGrid = maxGridList[levId];
         }
         for(nextLevId = (levId-1); nextLevId >=  0; nextLevId--){
            while(maxGridList[nextLevId] == -1 && nextLevId >= 0){//this does not hold max value, try for next grid
               nextLevId--;
            }           

            if(nextLevId >= 0){ //at least two max values found
               xSize = gridIdToSize[maxGridList[levId]].x;
               ySize = gridIdToSize[maxGridList[nextLevId]].y;

               intersectId = gridSizeToId[xSize - adjustIndexX][ySize - adjustIndexY];
               failedTest = (intersectId == -1)? 1: isGridFailed[intersectId];

               if(failedTest == 1){ //intersection is failed or invalid
                  continue;
               }
               else if(failedTest == 0){ //intersection is not failed or not invalid
                  coeffListAlt[maxGridList[levId]] = +1.0;
                  coeffListAlt[maxGridList[nextLevId]] = +1.0;
                  coeffListAlt[intersectId] = -1.0;
	
                  minusCoeffCounter++;	          

                  levId = nextLevId+1; // replace levId with nextLevId in outer loop (after decrement)
                  nextLevId = 0; //break; //breaking inner loop (condition will fail after the decrement in loop)
               }
            }//end of if(nextLevId < level...
         }//end of for(nextLevId = ...
      }//end of for(levId = level-1...
   }//end of if(totMax > 0 ...        

   // coeffList is the first priority
   if(totMax == 1 || minusCoeffCounter == 0){
      coeffList[onlyContribGrid] = +1.0;
   }

   // Choose the one which contains both the first and last grid of upper layer (if possible)
   // Copy the values
   if(coeffList[0] == 1 && coeffList[level-1] == 1){
      #pragma omp parallel for default(shared)
      for(gid = 0; gid < nGrids(); gid++){
         listCoeffs[gid] = coeffList[gid];
      }
   }
   else if(coeffListAlt[0] == 1 && coeffListAlt[level-1] == 1){
      #pragma omp parallel for default(shared)
      for(gid = 0; gid < nGrids(); gid++){
         listCoeffs[gid] = coeffListAlt[gid];
      }
   }
   else{
      #pragma omp parallel for default(shared)
      for(gid = 0; gid < nGrids(); gid++){
         listCoeffs[gid] = coeffList[gid];
      }
   }

   // Memory release 
   for(levId = 0; levId < level; levId++){
       delete[]  maxGridList2D[levId];
   }   
   delete[] maxGridList2D;
   
   delete[] isMax;
   delete[] coeffList;
   delete[] coeffListAlt;
   delete[] isGridFailed;
   delete[] maxGridList;
}//selectCombCoeffs()


