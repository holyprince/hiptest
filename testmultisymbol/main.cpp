
#include<stdio.h>
#include <hip/hip_runtime.h>
#define BLOCK_SIZE 64

__global__  void hip_normal(hipLaunchParm lp)
{

}


int main(int argc, char *argv[])
{
    hipLaunchKernel(hip_normal,10,10,0,0);
    return 0;
}



/*
template <typename T,typename X>
__global__  void hip_kernel_multi(
		T *A,
		T S,
		int image_size)
{
	int pixel = threadIdx.x + blockIdx.x*BLOCK_SIZE;
	if(pixel<image_size)
		A[pixel] = A[pixel]*S;
}

__global__  void hip_normal()
{
}


int main(int argc, char *argv[])
{
//	hipLaunchKernel(hip_normal,10,10,0,0);
//	return 0;
	int BSZ = ( (int) ceilf(( float) 128 /(float)BLOCK_SIZE));
    float *temp;
	hipMalloc((void **)&temp,500*sizeof(float));
    hipLaunchKernelGGL(hip_kernel_multi<float,int>,BSZ,BLOCK_SIZE,0,0,temp,2,128);
	
    return 0;
}*/

