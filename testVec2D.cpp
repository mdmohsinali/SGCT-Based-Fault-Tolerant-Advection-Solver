/* ParSGCT Code source file.
Copyright (c) 2015 Peter Edward Strazdins. All rights reserved.
Licensed under the terms of the BSD License as described in the LICENSE_SGCT file.
This comment must be retained in any redistributions of this source file.
*/

#include <stdio.h>
#include <stdlib.h>
#include "Vec2D.h"
int main(int argc, char** argv) {
  Vec2D<int> v = Vec2D<int>(2, 4);
  Vec2D<int> w;
  w.x = 1; w.y = 1;
  v = v + 1; v = v / w;
  printf("v=%d,%d=%d,%d\n", v.x, v.y, v.v[0], v.v[1]);
}
