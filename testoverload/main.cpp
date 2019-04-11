#include <assert.h>
#include <stdio.h>
#include <algorithm>
#include <stdlib.h>
#include<iostream>
#include "hip/hip_runtime.h"


#define WIDTH     20
#define HEIGHT    10

#define NUM       (WIDTH*HEIGHT)

#define THREADS_PER_BLOCK_X  5
#define THREADS_PER_BLOCK_Y  1
#define THREADS_PER_BLOCK_Z  1




__device__  void translatePixel(int a,int b,float c,float d,float &e,float &f)
{
}

__device__  void translatePixel(int a,int b,int ab,float c,float d, float cd,float &e,float &f,float &ef)
{

}





__global__ void 
vectoradd_float(hipLaunchParm lp, int width, int height) 

{
	int a,b;
	float c;
	translatePixel(a,b,c,c,c,c);
	translatePixel(a,b,a,c,c,c,c,c,c);
}

int main()
{
	 hipLaunchKernel(vectoradd_float,dim3(WIDTH/THREADS_PER_BLOCK_X, HEIGHT/THREADS_PER_BLOCK_Y),THREADS_PER_BLOCK_X,0, 0,WIDTH ,HEIGHT);
}
