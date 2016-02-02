/* ParSGCT Code source file.
Copyright (c) 2015 Peter Edward Strazdins. All rights reserved.
Licensed under the terms of the BSD License as described in the LICENSE_SGCT file.
This comment must be retained in any redistributions of this source file.
*/

// 2D vector class supporting basic arithmetic
// written by Peter Strazdins, May 13                                          

#ifndef VEC2D_INCLUDED
#define VEC2D_INCLUDED

#include <algorithm> // std::min
 
template <class T>
class Vec2D {
 public:
  union {
    struct {T x, y;};
    T v[2];
  };
  Vec2D<T> (T xv, T yv) {
    x = xv; y = yv;
  }
  Vec2D<T> () {
    x = y = 0;
  }
 
  Vec2D<T> min(Vec2D<T> v) {
    return Vec2D<T> (std::min(x, v.x), std::min(y, v.y));
  } 
  T prod() {
    return x * y;
  }

  Vec2D<T> operator + (const Vec2D<T>& v) const {  
    return Vec2D<T>  (x + v.x, y + v.y);
  }
  Vec2D<T> operator + (int v) const {  
    return Vec2D<T>  (x + v, y + v);
  }
  Vec2D<T> operator - (const Vec2D<T>& v) const {  
    return Vec2D<T>  (x - v.x, y - v.y);
  }
  Vec2D<T> operator - (int v) const {  
    return Vec2D<T>  (x - v, y - v);
  }
  Vec2D<T> operator * (const Vec2D<T>& v) const {  
    return Vec2D<T>  (x * v.x, y * v.y);
  }
  Vec2D<T> operator * (int v) const {  
    return Vec2D<T>  (x * v, y * v);
  }
  Vec2D<T> operator / (const Vec2D<T>& v) const {  
    return Vec2D<T>  (x / v.x, y / v.y);
  }
  friend Vec2D<T> operator / (int a, const Vec2D<T>& v) {  
    return Vec2D<T>  (a / v.x, a / v.y);
  }
  Vec2D<T> operator % (const Vec2D<T>& v) const {  
    return Vec2D<T>  (x % v.x, y % v.y);
  }
  friend bool operator == (const Vec2D<T>& u, const Vec2D<T>& v) {
    return (u.x == v.x && u.y == v.y);
  }
  friend bool operator <= (const Vec2D<T>& u, const Vec2D<T>& v) {
    return (u.x <= v.x && u.y <= v.y);
  }  
  
};

#endif /*VEC2D_INCLLDED*/
