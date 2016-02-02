/* ParSGCT Code source file.
Copyright (c) 2015 Peter Edward Strazdins. All rights reserved.
Licensed under the terms of the BSD License as described in the LICENSE_SGCT file.
This comment must be retained in any redistributions of this source file.
*/

// 2D Process Grid Class for handling 2D data grids distributed in a 
// block fashion.  
// Writeen by Peter Strazdins, May 13

#ifndef PROCGRID2D_INCLUDED
#define PROCGRID2D_INCLUDED

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <assert.h>

using namespace std;

class ProcGrid2D {
public:
  int myrank;    // this process's MPI rank /(=-1 if not part of this grid)
  int rank0;     // MPI rank of logical process (0.0) in this grid
  Vec2D<int> P;  // processor grid size
  Vec2D<int> id; // 2D rank in processor grid
  MPI_Comm comm;
  ProcGrid2D(int rank, int nprocs, MPI_Comm c, int r0=0) {
    P.x = P.y = 0;
    MPI_Dims_create(nprocs, 2, P.v);
    assert (P.x*P.y == nprocs);
    myrank = rank;
    rank0 = r0;
    comm = c;
    if (myrank == -1) 
      id.x = id.y = -1;
    else { 
      id.x = (myrank - rank0) % P.x;
      id.y = (myrank - rank0) / P.x;
    }
  }
  
  // get rank of process (pidx, pidy) in this grid
  int getRank(int pidx, int pidy) {
    assert (0 <= pidx && pidx < P.x && 0 <= pidy && pidy < P.y);
    return (rank0 + pidy*P.x + pidx);
  }

  // return global starting index in dim. d for global vector of length N
  // for this process 
  int L2G0(int d, int N) {
    return L2G0(d, N, id.v[d]);
  }
  // .. and for process index in pid
  int L2G0(int d, int N, int pid) {
    assert (0 <= d && d <= 1);  
    if (!(0 <= pid && pid < P.v[d])) {
      printf("%d: L2G0(%d,%d,%d) P=%d error\n", myrank, d, N, pid, P.v[d]);
      fflush(stdout); exit(1);
    }
    assert (0 <= pid && pid < P.v[d]);
    return ((N / P.v[d]) * pid); 
  }
  Vec2D<int> L2G0(Vec2D<int> N, Vec2D<int> pid) {
    return Vec2D<int>(L2G0(0, N.x, pid.x), L2G0(1, N.y, pid.y));
  }
  Vec2D<int> L2G0(Vec2D<int> N) {
    return L2G0(N, id);    
  } 
  
  // return local length corresponding to N in dim d in for this process 
  int G2L(int d, int N) {
    if (myrank == -1) // this process is not part of grid
      return 0;
    return G2L(d, N, id.v[d]);
  }
  // and for process index pid
  int G2L(int d, int N, int pid) {
    // printf("%d: G2L(%d, %d, %d), P=%d\n", myrank, d, N, pid, P.v[d]);
    assert (0 <= d && d <= 1);  
    /*
    if (!(0 <= pid && pid < P.v[d])) {
      int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      printf("%d: G2L(%d,%d,%d) P=%d ERROR!\n", rank, d, N, pid, P.v[d]);
      fflush(stdout);
    }
    assert (0 <= pid && pid < P.v[d]);
    */
    if (!(0 <= pid && pid < P.v[d])) {
      int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      printf("%d: G2L(%d,%d,%d) P=%d ERROR!\n", rank, d, N, pid, P.v[d]);
      fflush(stdout);
    }
    assert (0 <= pid && pid < P.v[d]);

    int n = N / P.v[d];
    if (pid == P.v[d]-1)
      n += N % P.v[d];
    return n;
  }
  Vec2D<int> G2L(Vec2D<int> N, Vec2D<int> pid) {
    return Vec2D<int>(G2L(0, N.x, pid.x), G2L(1, N.y, pid.y));
  }
  Vec2D<int> G2L(Vec2D<int> N) {
    if (myrank == -1) // this process is not part of grid
      return Vec2D<int>(0, 0);
    return G2L(N, id);    
  } 

  // returns whether process pid is last in dimension d (has extra elements)
  Vec2D<int> lastProc() {
    return lastProc(id);
  }
  int lastProc(int d, int pid) {
    return (P.v[d]-1 == pid);
  }
  Vec2D<int>  lastProc(Vec2D<int> pid) {
    return Vec2D<int>(lastProc(0, pid.x), lastProc(1, pid.y));
  }


  // return process index for element N0 in dimension d corresp. to N
  int getP0(int d, int N0, int N) {
    assert (0 <= d && d <= 1);
    assert (0 <= N0 && N0 <= N);
    // it is possible, e.g. with P=7 N=33 N0=28, that the last process 
    // has >= 2x the points of others
    return (std::min(N0 / (N / P.v[d]), P.v[d]-1));
  }
  Vec2D<int> getP0(Vec2D<int> N0, Vec2D<int> N) {
    return Vec2D<int>(getP0(0, N0.x, N.x), getP0(1, N0.y, N.y));
  } 

  // return offset in process for element N0 in dimension d corresp. to N
  int getOffs0(int d, int N0, int N) {
    assert (0 <= d && d <= 1);
    assert (0 <= N0 && N0 <= N);
    return (N0 % (N / P.v[d]));
  }
  Vec2D<int> getOffs0(Vec2D<int> N0, Vec2D<int> N) {
    return Vec2D<int>(getOffs0(0, N0.x, N.x), getOffs0(1, N0.y, N.y));
  } 

  // returns if process pid in grid owns points N0..N0+dN-1. for grid length N 
  bool ownsData(int d, int pid, int N0, int  dN, int N) {
    return (L2G0(d, N, pid) <= N0 && N0+dN <= L2G0(d, N, pid)+G2L(d, N, pid)); 
  }
  bool ownsData(Vec2D<int> pid, Vec2D<int> N0, Vec2D<int> dN, Vec2D<int> N) {
    return (ownsData(0, pid.x, N0.x, dN.x, N.x) && 
	    ownsData(1, pid.y, N0.y, dN.y, N.y));
  }
  
  // return rank of process in direction dir and dimension d 
  // left: dir=-1,d=0: right: dir=+1,d=0; up: dir=-1,d=1; down: dir=+1,d=1 
  int neighbour(int dir, int d) {
    assert (0 <= d && d <= 1);  
    assert (dir==-1 || dir==1);  
    Vec2D<int> nid;
    nid.v[d] = (P.v[d] + id.v[d] + dir) % P.v[d];
    nid.v[1-d] = id.v[1-d];
    return (getRank(nid.x, nid.y));
  }
};

#endif /*PROCGRID2D_INCLUDED*/
