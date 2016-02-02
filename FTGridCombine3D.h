/* SGCT-Based-Fault-Tolerant-Advection-Solver Code source file.
   Copyright (c) 2015, Md Mohsin Ali. All rights reserved.
   Licensed under the terms of the BSD License as described in the LICENSE_FT_CODE file.
   This comment must be retained in any redistributions of this source file.
*/

// Parallel 2D Sparse Grid Combination class
// written by Peter Strazdins, May 13
// updated by Mohsin Ali, September 2013, February 2014

#include "Vec3D.h"
#include "Timer.h"
#include "ProcGrid3D.h"
#include "FTHaloArray3D.h"
#include <stdlib.h>     /* abs() */

#define CLASSIC_COMB              0
#define RESAMP_AND_COPY           1
#define ALTERNATE_COMB            2
#define CHKPOINT_RESTART          3

class GridCombine2D {
  public:
    
  // return size corresponding to co-ordinate gix
  static int gridSz(int gix) {
    return (1 + (1 << gix));
  }
  // return size vector corresponding to co-ordinates gix
  static Vec2D<int> gridSz(Vec2D<int> gix) {
    Vec2D<int> g;
    g.x = 1 + (1 << gix.x);
    g.y = 1 + (1 << gix.y);
    return (g);
  }
  // variant ignoring redundant points on bottom right boundaries
  // This corresponds to the scaling of grid points
  static Vec2D<int> gridSz1(Vec2D<int> gix) {
    Vec2D<int> g;
    g.x = (1 << gix.x);
    g.y = (1 << gix.y);
    return (g);
  }

  // return default value of number of processes on 2nd diagonal 
  // (to achieve load balanced), for the middle grids
  static int getPD1(int pD0, bool fixedProcs) {
    return ((pD0+1)/2);
  }

  // return default value of number of processes on 3rd diagonal 
  // (to achieve load balanced), for the lower grids
  static int getPD2(int pD1, bool fixedProcs) {
    return ((pD1+1)/2);
  }

  // return number of processes at grid of index (i, level-i)
  static int nProcsPt(int level, int pD, bool fixedProcs, int i) {
    if (fixedProcs)
      return pD;
    int shift = (level%2 == 1)? abs(i - level/2): 
      (i < level/2)? level/2 - i - 1: i - level/2; //ugh
    // printf("nProcsPt(%d,%d,1,%d): shift=%d\n", level, pD, i, shift);
    return (pD * (1 << shift));
  }
  
  // return total number of processes required in a diagonal
  static int nProcsDiag(int level, int pD, bool fixedProcs) {
    if (fixedProcs)
      return(level*pD);
    int nP = 0;
    for (int i=0; i < level; i++) 
      nP += nProcsPt(level, pD, fixedProcs, i);
    return (nP);
  }

  static int nProcs(int version, int level, int pD[2], bool fixedProcs) {
     int v = 0;
     if(version == CLASSIC_COMB || version == CHKPOINT_RESTART){
        v = (nProcsDiag(level, pD[0], fixedProcs) + 
	    nProcsDiag(level-1, pD[1], fixedProcs));  
     }
     else if(version == RESAMP_AND_COPY){
        v = (2 * (nProcsDiag(level, pD[0], fixedProcs)) + 
	      nProcsDiag(level-1, pD[1], fixedProcs));
     }
     else if(version == ALTERNATE_COMB){
        v = (nProcsTwoDiags(level, pD, fixedProcs) + nProcsDiag(level-2, pD[1]/2, fixedProcs) + 
             nProcsDiag(level-3, pD[1]/4, fixedProcs));
     }
    
     return (v);
  }

  static int nProcsTwoDiags(int level, int pD[2], bool fixedProcs) {
    return (nProcsDiag(level, pD[0], fixedProcs) + 
	    nProcsDiag(level-1, pD[1], fixedProcs)); 
  }



  // get max. process diagonal size for nprocs total processes
  static int getPD0(int nprocs, int level, bool fixedProcs) {
    if (fixedProcs) 
      return (2*nprocs / (3*level-1));
    assert(false /*TODO*/); 
    return (0);
  }

  // return number of grids
  int nGrids();
  
  // return number of grids that are on upper and middle
  int nGridsTwoDiags();  

  // return number of grids of three diagonals
  int nGridsThreeDiags();  

  // return which grid id global rank is in
  int getGid(int rank);
    
  // return rank within grid for process with global rank
  int getGrank(int rank);

  // return co-ordinates of grid gid
  Vec2D<int> gridIx(int gid);

  // return diagonal grid gid is on
  int diagRank(int gid);

  // return index within diagonal grid gid is on
  int diagIx(int gid);
  
  // return pgs for the input grid Id gid
  ProcGrid2D* getPgs(int gid){
      return pgs[gid];
  }  

 private:
  int pD[2], level, verbosity, version; bool debug;
  Vec2D<int> gridSize;  
  ProcGrid2D** pgs; // array of process grids for each component grid
  Timer *timer;
  int nprocsSg;     // total number of processes in separate sparse grid; 0 for in-sub-grids combine
  int tempRank;     // -1 if rank >= nprocs
  int nprocs;       // total number of processes in pgs; only sub-grids in case of separate sparse grid
  int myrank;       // global rank of this process; useful for debugging
  bool fixedProcs;  // whether number of process is fixed across diagonals
                    // in the grid index space. Otherwise it doubles as you
                    // go outwards from the middle
  int *gRanks, *gIds; //caches each process' rank within grid and grid index
  MPI_Comm myCommWorld; // communicator

  int  nGatherSends;  // number of GatherSends occur from this component grid 
  int  nGatherRecvs;  // number of GatherRecvs occur to this component grid  
  
  int  nScatterSends;  // number of ScatterSends occur from this component grid   
  int  nScatterRecvs;  // number of ScatterRecvs occur to this component grid   
   
  
  MPI_Request * nGatherSendRequests, * nGatherRecvRequests;
  MPI_Status * nGatherSendStatuses, * nGatherRecvStatuses;  
  
  MPI_Request * nScatterSendRequests, * nScatterRecvRequests;
  MPI_Status * nScatterSendStatuses, * nScatterRecvStatuses;
  
  double ** gatherSendBuff;
  double ** scatterSendBuff;
  double * gatherRecvCoeff;
  
  Vec2D<int> * grSU, * gdn, *sdn;
  int *gi, *gj, *si, *sj;
  HaloArray2D ** guR, **suR;

  int ** gridSizeToId;
  Vec2D<int> * gridIdToSize; 
  
  int gatherSendCounter, gatherRecvCounter;
  int scatterSendCounter, scatterRecvCounter; 

 public:
  Vec2D<int> gxU; //this process' component grid's co-ordinates
  ProcGrid2D* pgU; // this process' component grid's process grid

  Vec2D<int> gxS;  // co-ordinates for sparse grid
  ProcGrid2D* pgS; // process grid for sparse grid

  // create object for performing combination grids at level level
  // with p process on the outer diagonal
  /*
  GridCombine2D(int level, int pD[2], bool fixedProcs, int sgProcs, 
		Vec2D<int> gridS, Timer *timer, int verbosity=0, 
		bool dbg = false);
  */
  GridCombine2D(int level, int pD[2], bool fixedProcs, int sgProcs, 
		Vec2D<int> gridS, Timer *timer, int verbosity=0, 
		bool dbg = false, bool isSeparate = false, int ver = 0, MPI_Comm comm = MPI_COMM_WORLD);

  void GridCombine2DClassicComb(int sgProcs, Vec2D<int> gridS, bool isSeparate = false);
  void GridCombine2DResampAndCopy(int sgProcs, Vec2D<int> gridS, bool isSeparate = false);
  void GridCombine2DAlternateComb(int sgProcs, Vec2D<int> gridS, bool isSeparate = false);

  ~GridCombine2D();

  // returns this process'  component grid's process grid object
  ProcGrid2D* myProcGrid();

  // return's this process' component grid's process grid communicator
  MPI_Comm myProcGridCOMM(); 

  // perform the whole operation
  //void gatherScatter(HaloArray2D* u, HaloArray2D* usg);
  void gatherScatter(HaloArray2D* u, HaloArray2D* usg, int * failedList, int numFailed, MPI_Comm comm);

  // set up number of messages to be sent during gather and received during scatter
  void gatherSendScatterRecvInit();

  // set up number of messages to be received during gather and sent during scatter
  void gatherRecvScatterSendInit();
 

  // perform the operation of gather and scatter 
  void gatherScatterAllVersions(HaloArray2D* u, HaloArray2D* usg, int * failedList, int numFailed, MPI_Comm comm);

  // wait for buffers to be used and delete them
  //void gatherSendWait();
  void gatherSendWait(MPI_Comm comm);
  // current process sends its part of component grid (U) to the respective 
  // sparse grid (S) process
  void gatherSend(HaloArray2D* u);
  // and receive it back from the scatter
  void scatterRecv(HaloArray2D* u); 

  // wait for buffers to be used and delete them
  //void scatterSendWait();
  void scatterSendWait(MPI_Comm comm);

  // for each component grid in pgs, gather contributions from 
  // respective processes to the sparse grid (S)
  void gatherRecv(HaloArray2D* uS, int * failedList, int numFailed);
  // and scatter them back out
  void scatterSend(HaloArray2D* uS, int* failedList, int numFailed);

  void selectCombCoeffs(int * failedList, int numFailed, double * listCoeffs);

private:
  double *selfBuf;   // used to buffer data sent to self on these functions
  char *gatherSendBuf, *scatterSendBuf;   
  long int gatherSendBufSz, scatterSendBufSz; 

  // helper functions to reduce common code duplication
  void gatherSendScatterRecv(bool send, HaloArray2D* u);
  void gatherRecvScatterSend(bool recv, HaloArray2D* uS, int* failedList, int numFailed);
};


