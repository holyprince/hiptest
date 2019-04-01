#ifndef ADD_H
#define ADD_H
#include <stdio.h>
#include <stdlib.h>
#include "hip/hip_runtime.h"

__global__ void vectoradd_float(hipLaunchParm lp);

#endif
