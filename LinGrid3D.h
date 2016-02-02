/* ParSGCT Code source file.
Copyright (c) 2015 Peter Edward Strazdins. All rights reserved.
Licensed under the terms of the BSD License as described in the LICENSE_SGCT file.
This comment must be retained in any redistributions of this source file.
*/

// Linear Grid class; used for testing grid combination algorithn
// written by Peter Strazdins, May 13

#include  <cmath>
#include "FTHaloArray3D.h"
#include "ProcGrid3D.h"

class LinGrid {
  public:
  Vec2D<int> gridSize; /* number of points in grid */
  Vec2D<int> scaleR; 

  LinGrid(Vec2D<int> gSize, Vec2D<int> sR) {
    gridSize = gSize;
    scaleR = sR;
  }

  ~LinGrid(){}

  void initGrid(HaloArray2D *u, ProcGrid2D *g) {
    for (int j=0; j < u->l.y+1; j++) {
      double y = scaleR.y * (j + g->L2G0(1, gridSize.y));
      for (int i=0; i < u->l.x+1; i++) {
	double x = scaleR.x * (i + g->L2G0(0, gridSize.x));
	Vh(u, i, j) = x + y;
      }
    }
  }

  double checkError(double scale, HaloArray2D *u, ProcGrid2D *g) {
    double err = 0.0;
    for (int j=0; j < u->l.y; j++) {
      double y = scaleR.y * (j + g->L2G0(1, gridSize.y));
      for (int i=0; i < u->l.x; i++) {
	double x = scaleR.x  * (i + g->L2G0(0, gridSize.x));
	err += std::abs(Vh(u, i, j) - scale*(x + y));
      }
    }
    return (err);
  }
};

