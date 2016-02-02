/* SGCT-Based-Fault-Tolerant-Advection-Solver Code source file.
   Copyright (c) 2015, Md Mohsin Ali. All rights reserved.
   Licensed under the terms of the BSD License as described in the LICENSE_FT_CODE file.
   This comment must be retained in any redistributions of this source file.
*/

// Simple 2D (x,y)  array with a halo class. 
// Storage is contiguous the x co-ordinate 
// written by Peter Strazdins, May 13

#ifndef HALOARRAY2D_INCLUDED
#define HALOARRAY2D_INCLUDED

#include <stdio.h>
#include <assert.h>
#include <cmath>  //std::abs, log2 
#include <string> //std::string
#ifdef _OPENMP
#include <omp.h>
#endif
#include "Vec3D.h"

class HaloArray2D {
public:
  double *u;
  Vec2D<int> l, s; // local size, storage size (= local + halo) 
  int halo;

  HaloArray2D(int lx, int ly, int h) {
    l.x = lx;  l.y = ly;
    assert (h >= 0);
    halo = h;
    s = l + 2*halo;
    // check for potential overflow; should change to Vec2D<long> in future 
    assert(log2(s.x) + log2(s.y) < sizeof(int)*8 - 1);
    if (s.x*s.y > 0)
      u = new double[s.x*s.y];
    else
      u = 0;
  }    

  HaloArray2D(Vec2D<int> l_, int h) {
   // note: we may not use HaloArray2D(l_.x, l_.y,h) 
    l = l_;
    assert (h >= 0);
    halo = h;
    s = l + 2*halo;
    // check for potential overflow; should change to Vec2D<long> in future 
    assert(log2(s.x) + log2(s.y) < sizeof(int)*8 - 1);
    if (s.x*s.y > 0)
      u = new double[s.x*s.y];
    else
      u = 0;
  }

  ~HaloArray2D() {
    if (u != 0)
      delete[] u;
  }

  inline double *ix(int i, int j) {
    // assert (0 <= i && i < s.x  &&  0 <= j && j < s.y);
    return(&u[i + (j)*s.x]);
  }

  inline double *ix_h(int i, int j) {
    return ix(i + halo, j + halo);
  }

  void zero() {
    #pragma omp parallel for default(shared)  
        for (int j=0; j < l.y; j++){ 
          for (int i=0; i < l.x; i++) {
            *ix_h(i, j) = 0.0;
          }
        }  
  }

  double norm1() {
    double norm = 0.0;
    for (int j=0; j < l.y; j++) 
      for (int i=0; i < l.x; i++) {
        norm += std::abs(*ix_h(i, j));
      }
    return (norm);
  }

  double *pack(int i0, int j0, int m, int n) {    
    double* buf = new double [m*n];
    #pragma omp parallel for default(shared)
        for (int j=0; j < n; j++) {
          double *b = &buf[j * m]; 
          for (int i=0; i < m; i++) {
            *b = *ix_h(i+i0, j+j0);
            b++;
          }
        }
    return (buf); 
  }

  void unpack(double* buf, int i0, int j0, int m, int n) {
  #pragma omp parallel for default(shared)
    for (int j=0; j < n; j++) {
      double *b = &buf[j * m];
      for (int i=0; i < m; i++) {
        *ix_h(i+i0, j+j0) = *b;
	b++;
      }
    }
  }

  void interpolate(double coeff, HaloArray2D *v, Vec2D<int> rV,
		   int i0, int j0, Vec2D<int>n) {
    #pragma omp parallel for default(shared)  
        for (int j=0; j < n.y; j++) {
          double y = (1.0 * j) / rV.y;
          int iy = std::min((int) y, n.y-1);
          double ry = y - iy;
          for (int i=0; i < n.x; i++) {
            double x = (1.0 * i) / rV.x;
            int ix = std::min((int) x, n.x-1);
            double rx = x - ix;
            double z =
              (1.0-rx)*(1.0-ry)* *(v->ix_h(ix, iy)) + 
              rx      *(1.0-ry)* *(v->ix_h(ix+1, iy)) +
              (1.0-rx)*ry      * *(v->ix_h(ix, iy+1)) + 
              rx      *ry      * *(v->ix_h(ix+1, iy+1));
              *ix_h(i+i0, j+j0) += coeff * z;
          }
        }
  } //interpolate()
 
  double *sample(int i0, int j0, Vec2D<int> rV, Vec2D<int>n) {
    double *buf = new double [n.prod()];
    #pragma omp parallel for default(shared)
         for (int j=0; j < n.y; j++) {
          double *v = &buf[j * n.x];
          for (int i=0; i < n.x; i++) {
            *v = *ix_h(i0 + i*rV.x, j0 + j*rV.y);
            *v = *ix_h(i0 + i*rV.x, j0 + j*rV.y);
            v++;
          }
        }
    return buf;
  } //sample()
 
  void print(int rank, std::string label) {
    if (label.c_str()[0] != 0)
      printf("%d: %s:\n", rank, label.c_str());
    for (int j=0; j < l.y; j++) {
      if (rank >= 0)
	printf("%d: ", rank);
      for (int i=0; i < l.x; i++) 
        printf("(%d,%d)%+0.2f ", i, j, *ix_h(i,j));
      printf("\n"); 
    }
    printf("\n"); 
  }


  void printh(int rank, std::string label) {
    if (label.c_str()[0] != 0)
      printf("%d: %s:\n", rank, label.c_str());
    for (int j=0; j < s.y; j++) {
      if (rank >= 0)
	printf("%d: ", rank);
      for (int i=0; i < s.x; i++) 
        printf("%+0.2f ", *ix(i,j));
      printf("\n"); 
    }
    printf("\n"); 
  }
};

// macros for access elements for logical array wihtout the halo
#define V(u, i, j) (*((u)->ix(i, j)))
// macros for accessing elements taking into account the halo
#define Vh(u, i, j) (*((u)->ix_h(i, j)))

#endif /*HALOARRAY2D_INCLUDED*/
