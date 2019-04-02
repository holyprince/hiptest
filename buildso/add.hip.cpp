#include "add.h"








__global__ void vectoradd_float_kernel   (hipLaunchParm lp)

{

}


void vectoradd_float(int blocks, int threads){
	    hipLaunchKernel(vectoradd_float_kernel,blocks,threads,0, 0);
}
