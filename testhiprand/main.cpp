#include <stdio.h>


#include <hip/hip_runtime.h>

#include <hiprand_kernel.h>
#include <hiprand.h>



__global__ void hip_kernel_randtest(hipLaunchParm lp)
{

	int i=hipThreadIdx_x;
	hiprandState_t state;
	hiprand_init(1234, 100, 0, &state);
	int value=hiprand_normal2(&state).x;
}





int main()
{
	hipLaunchKernel(hip_kernel_randtest, dim3(1),dim3(100),0, 0);
	return 0;
}
