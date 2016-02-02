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
#include <cmath> // M_PI
#include "FTHaloArray3D.h"
#include "ProcGrid3D.h"
#include "Timer.h"

class Advect2D {
  public:
  double tf /*final time*/, 
         dt /*timestep*/;
  Vec2D<double> delta /*grid spacing*/, V /*advection velocity*/;
  Vec2D<int> gridSize /* number of points in grid*/;
  int method; // 1st or 2nd order method}
  int verbosity; // for debugging output
  Timer *timer;

  MPI_Comm subComm, myComm, myCommWorld;
  int gId, gRank;
  

  Advect2D(Vec2D<int> gridSz, Vec2D<double> v, double timef, double CFL, 
	   int meth, int verb, Timer *timer, MPI_Comm SubSparseComm, MPI_Comm selfComm, int gridId, int gridRank, MPI_Comm myCW); 

  ~Advect2D(){}

  // update u for a timestep by various schemes
  void updateGodunov(HaloArray2D *u);
  void updateLW(HaloArray2D *u);  
  void updateMacCormack(HaloArray2D *u);

  inline double initialCondition(double x, double y, double t=0.0,
                         double vx=1.0, double vy=1.0) {
    x = x - vx*t;
    y = y - vy*t;
    return std::sin(4.0*M_PI*x) * std::sin(2.0*M_PI*y);
  }

  void initGrid(HaloArray2D *u, ProcGrid2D *g);

  double checkError(double t, HaloArray2D *u, ProcGrid2D *g);

  void updateBoundary(HaloArray2D *u, ProcGrid2D *g, int test);

  double simulateAdvection(HaloArray2D *u, ProcGrid2D *g);

};

