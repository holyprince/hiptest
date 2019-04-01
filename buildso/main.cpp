#include <stdio.h>
#include <stdlib.h>
#include "hip/hip_runtime.h"
#include "add.h"


#define WIDTH     20
#define HEIGHT    10

#define NUM       (WIDTH*HEIGHT)

#define THREADS_PER_BLOCK_X  5
#define THREADS_PER_BLOCK_Y  1
#define THREADS_PER_BLOCK_Z  1




int main() {
  
  hipLaunchKernel(vectoradd_float, 
                  dim3(WIDTH/THREADS_PER_BLOCK_X, HEIGHT/THREADS_PER_BLOCK_Y),
                  THREADS_PER_BLOCK_X,0, 0);

  return 0;
}
