/* SGCT-Based-Fault-Tolerant-Advection-Solver Code source file.
   Copyright (c) 2015, Md Mohsin Ali. All rights reserved.
   Licensed under the terms of the BSD License as described in the LICENSE_FT_CODE file.
   This comment must be retained in any redistributions of this source file.
*/

// Parallel 2D Advection Class
// based on codes by Brendan Harding
// written by Peter Strazdins, May 13
// modified by Mohsin Ali, August 13

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <assert.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "FTAdvect3D.h"

Advect2D::Advect2D(Vec2D<int> gridSz, Vec2D<double> v, 
		   double timef, double CFL, int meth, int verb, Timer *t, MPI_Comm SubSparseComm, MPI_Comm selfComm, int gridId, int gridRank, MPI_Comm myCW) { 

  subComm = SubSparseComm;
  myComm = selfComm;
  gId = gridId;
  gRank = gridRank;
  myCommWorld = myCW;

  gridSize = gridSz;
  V = v; 
  delta.x = 1.0 / (gridSize.x - 1);  // implicitly on unit square    
  delta.y = 1.0 / (gridSize.y - 1);
  tf = timef;
  dt = CFL * std::min<double>(delta.x, delta.y);
  method = meth;
  verbosity = verb;
  timer = t;
}

void Advect2D::updateGodunov(HaloArray2D *u) {
  timer->start("updateGodunov", u->l.prod(), 1);
  HaloArray2D * f_x = new HaloArray2D(u->l.x, u->l.y, 0);  
  HaloArray2D * f_y = new HaloArray2D(u->l.x, u->l.y, 0);  
  int fdx = (V.x >= 0.0)? -1: +1,
      fdy = (V.y >= 0.0)? -1: +1;
  #pragma omp parallel for default(shared)  
  for (int j=0; j < u->l.y; j++){     
     for (int i=0; i < u->l.x; i++) {
        V(f_x, i, j) = V.x * (Vh(u, i, j) - Vh(u, i+fdx, j));
        V(f_y, i, j) = V.y * (Vh(u, i, j) - Vh(u, i, j+fdy));
      }
  }  
  double dtdx = -fdx * dt / delta.x, dtdy = -fdy * dt / delta.y;
  #pragma omp parallel for default(shared)  
  for (int j=0; j < u->l.y; j++){
     for (int i=0; i < u->l.x; i++) {
        Vh(u, i, j) += - dtdx * V(f_x, i, j) - dtdy * V(f_y, i, j);
     }
  }  
  delete f_x;
  delete f_y;
  timer->stop("updateGodunov");
} //updateGodunov()

void Advect2D::updateLW(HaloArray2D *u) {
  timer->start("updateLW", u->l.prod(), 1);
  HaloArray2D * uh = new HaloArray2D(u->s.x-1, u->s.y-1, 0);  
  double sx = 0.5 * V.x / delta.x, sy = 0.5 * V.y / delta.y;
  #pragma omp parallel for default(shared)
  for (int j=0; j < uh->l.y; j++){     
     for (int i=0; i < uh->l.x; i++) {
        V(uh,i,j) = 0.25*(Vh(u,i,j) + Vh(u,i-1,j) + Vh(u,i,j-1) + Vh(u,i-1,j-1))
          -0.5*dt*(sx*(Vh(u,i,j) + Vh(u,i,j-1) - Vh(u,i-1,j) - Vh(u,i-1,j-1)) +
                   sy*(Vh(u,i,j) + Vh(u,i-1,j) - Vh(u,i,j-1) - Vh(u,i-1,j-1)));
     }
  } 
 
  double dtdx = 0.5 * dt / delta.x, dtdy = 0.5 * dt / delta.y;
  #pragma omp parallel for default(shared)
  for (int j=0; j < u->l.y; j++){     
     for (int i=0; i < u->l.x; i++) {
        Vh(u, i, j) += - dtdx * (V(uh,i+1,j+1) + V(uh,i+1,j) - V(uh,i,j) - V(uh,i,j+1))
         - dtdy * (V(uh,i+1,j+1) + V(uh,i,j+1) - V(uh,i,j) - V(uh,i+1,j));
     }
  } 

  delete uh;
  timer->stop("updateLW");
} //updateLW()

void Advect2D::updateMacCormack(HaloArray2D *u) {
  timer->start("updateMacCormack", u->l.prod(), 1);
  HaloArray2D * up = new HaloArray2D(u->s.x-1, u->s.y-1, 0);  
  double sx = V.x * dt / delta.x, sy = V.y * dt / delta.y;
  #pragma omp parallel for default(shared)
  for (int j=0; j < up->l.y; j++){     
     for (int i=0; i < up->l.x; i++) {
        V(up,i,j) = Vh(u,i-1,j-1) - sx * (Vh(u,i,j-1) - Vh(u,i-1,j-1)) 
            - sy * (Vh(u,i-1,j) - Vh(u,i-1,j-1));	
        }
  } 
  sx = 0.5 * sx; sy = 0.5 * sy; 
  #pragma omp parallel for default(shared)
  for (int j=0; j < u->l.y; j++){     
     for (int i=0; i < u->l.x; i++) {
        Vh(u, i, j) = 0.5 * (Vh(u, i, j) + V(up, i+1, j+1))
                           - sx * (V(up, i+1, j+1) - V(up, i, j+1))
                           - sy * (V(up, i+1, j+1) - V(up, i+1, j));
     }
  } 
  delete up;
  timer->stop("updateMacCormack");
}

void Advect2D::initGrid(HaloArray2D *u, ProcGrid2D *g) {
  #pragma omp parallel for default(shared)  
  for (int j=0; j < u->l.y; j++) {
     double y = delta.y  * (j + g->L2G0(1, gridSize.y));
     for (int i=0; i < u->l.x; i++) {
        double x = delta.x  * (i + g->L2G0(0, gridSize.x));
        Vh(u, i, j) = initialCondition(x, y, 0.0, V.x, V.y);
      }
  }
}

double Advect2D::checkError(double t, HaloArray2D *u, ProcGrid2D *g) {
  double err = 0.0;
  #pragma omp parallel for default(shared) reduction(+:err)
  for (int j=0; j < u->l.y; j++) {
     double y = delta.y  * (j + g->L2G0(1, gridSize.y));
     for (int i=0; i < u->l.x; i++) {
        double x = delta.x  * (i + g->L2G0(0, gridSize.x));
           err += std::abs(Vh(u, i, j) - initialCondition(x, y, t, V.x, V.y));
     } 
  }
  return (err);
}

// values at grid rows/columns 0 and N-1 are indentical; therefore  
// boundaries must get their value the next innermost opposite row/column
void Advect2D::updateBoundary(HaloArray2D *u, ProcGrid2D *g, int test) {
  assert(u->halo == 1); 
  timer->start("updateBoundary", u->l.prod(), 1);
  int lx = u->l.x, ly = u->l.y, sx = u->s.x; 
  MPI_Request req; MPI_Status stat;

  if (g->P.x == 1) {  
    for (int j=1; j < ly+1; j++) 
      V(u, 0, j)    = V(u, lx-1, j);
    for (int j=1; j < ly+1; j++) 
      V(u, lx+1, j) = V(u, 2, j);
  } else {
    double *bufS = new double[ly], *bufR = new double[ly]; 

    int xOffs = (g->id.x == g->P.x-1) ? u->l.x-1: u->l.x;
    for (int j=1; j < ly+1; j++) 
      bufS[j-1] = V(u, xOffs, j);
    //printf("%d: comm right boundary to %d\n",g->myrank,g->neighbour(+1,0));
     MPI_Isend(bufS, ly, MPI_DOUBLE, g->neighbour(+1, 0), 0, g->comm, &req);
     MPI_Recv(bufR, ly, MPI_DOUBLE, g->neighbour(-1, 0), 0, g->comm, &stat);
     for (int j=1; j < ly+1; j++)
       V(u, 0, j) = bufR[j-1];
     MPI_Wait(&req, &stat);

    xOffs = (g->id.x == 0) ? 2: 1;
    for (int j=1; j < ly+1; j++) 
      bufS[j-1] = V(u, xOffs, j);
    //printf("%d: comm left boundary to %d\n",g->myrank,g->neighbour(-1,0));
     MPI_Isend(bufS, ly, MPI_DOUBLE, g->neighbour(-1, 0), 0, g->comm, &req);
     MPI_Recv(bufR, ly, MPI_DOUBLE, g->neighbour(+1, 0), 0, g->comm, &stat);
     for (int j=1; j < ly+1; j++)
       V(u, lx+1, j) = bufR[j-1];
     MPI_Wait(&req, &stat);

     delete[] bufS; delete[] bufR;
  }
   
  if (g->P.y == 1) {
    for (int i=0; i < sx; i++)
      V(u, i, 0)    = V(u, i, ly-1);
    for (int i=0; i < sx; i++)
      V(u, i, ly+1) = V(u, i, 2);
  } else {
    double *bufS = new double[sx], *bufR = new double[sx]; 

    int yOffs = (g->id.y == g->P.y-1) ? ly-1: ly;
    for (int i=0; i < sx; i++) 
       bufS[i] = V(u, i, yOffs);
     //printf("%d: comm bottom boundary to %d\n",g->myrank,g->neighbour(+1,1));
     MPI_Isend(bufS, sx, MPI_DOUBLE, g->neighbour(+1, 1), 0, g->comm, &req);
     MPI_Recv(bufR, sx, MPI_DOUBLE, g->neighbour(-1, 1), 0, g->comm, &stat);
     for (int i=0; i < sx; i++)
       V(u, i, 0) = bufR[i];
     MPI_Wait(&req, &stat);

    yOffs = (g->id.y == 0) ? 2: 1;
    for (int i=0; i < sx; i++) 
       bufS[i] = V(u, i, yOffs);
     //printf("%d: comm top boundary to %d\n", g->myrank, g->neighbour(-1, 1));
     MPI_Isend(bufS, sx, MPI_DOUBLE, g->neighbour(-1, 1), 0, g->comm, &req);
     MPI_Recv(bufR, sx, MPI_DOUBLE, g->neighbour(+1, 1), 0, g->comm, &stat);
     for (int i=0; i < sx; i++)
        V(u, i, ly+1) = bufR[i];
     MPI_Wait(&req, &stat);

     delete[] bufS; delete[] bufR;
  }
  timer->stop("updateBoundary");
} //updateBoundary()

double Advect2D::simulateAdvection(HaloArray2D *u, ProcGrid2D *g) {

    double t = 0.0;  
    int s = 0;

    if (method == 0){
      updateGodunov(u);
    }
    else if (method == 1){
      updateLW(u); 
    }
    else{
      updateMacCormack(u);
    }
    updateBoundary(u, g, 0);

    t += dt; s++;

//  int rank;
//  MPI_Comm_rank(g->comm, &rank);
//  printf("%d: simulateAdvection: s=%d, dt=%e  t=%e\n", rank, s, dt, t);

  return t;
} //simulateAdvection()
