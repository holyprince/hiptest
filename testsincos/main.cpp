#include <stdio.h>
#include <hip/hip_runtime.h>
#define HIP_ENABLE_PRINTF
#define HCC_ENABLE_PRINTF

#define WIDTH     20
#define HEIGHT    10

#define NUM       (WIDTH*HEIGHT)

#define THREADS_PER_BLOCK_X  5
#define THREADS_PER_BLOCK_Y  1
#define THREADS_PER_BLOCK_Z  1




__global__ void 
vectoradd_float() 

{
	float a=3.14;
	float sina,cosa;
	sincosf(a,&sina,&cosa);
	printf("%f ",cosa);
}


int main()
{
	float *da;
	float *db;
	hipMalloc((void**)&da, NUM * sizeof(float));
	hipMalloc((void**)&db, NUM * sizeof(float));
	hipLaunchKernelGGL(vectoradd_float, dim3(WIDTH/THREADS_PER_BLOCK_X, HEIGHT/THREADS_PER_BLOCK_Y),THREADS_PER_BLOCK_X,0,0);
}
