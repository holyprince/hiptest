
#include <iostream>
using namespace std;
#include <hip/hip_runtime.h>
#include <hiprand.h>
#include <hiprand_kernel.h>
#include <rocrand_kernel.h>

__global__ void hip_kernel_randtest()
{
	hiprandState States;
	float2 data1;
	rocrand_init(1234, 100, 0, &States);
	data1= rocrand_normal2(&States);
}

__global__ void hip_kernel_randtest2()
{
	hiprandStateXORWOW_t States;
	float2 data1;
	hiprand_init(1234, 100, 0, &States);
	data1= hiprand_normal2(&States);
}





int main(int argc, char *argv[])
{

	hipLaunchKernelGGL(hip_kernel_randtest,128,128,0,0);
	hipLaunchKernelGGL(hip_kernel_randtest2,128,128,0,0);
    return 0;
}
