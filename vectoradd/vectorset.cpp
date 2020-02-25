
//#include <iostream>
//using namespace std;
#include <stdio.h>
#include <hip/hip_runtime.h>
#include "hip/hip_runtime_api.h"
#include "hip/hip_vector_types.h"
#include "rocfft.h"


#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

#define NX 10
#define NY 10
#define NZ 1
#define NXYZ NX*NY*NZ

static void HandleError( hipError_t err, const char *file, int line )
{

    if (err != hipSuccess)
    {
    	fprintf(stderr, "ERROR: %s in %s at line %d (error-code %d)\n",
						hipGetErrorString( err ), file, line, err );
		fflush(stdout);
    }
}
void datainit(float2 *data)
{
        for (int i = 0; i < NXYZ; i++) {
                data[i].x = i % 1000 ;
                data[i].y= 0;
        }
}

void datainitzero(float2 *data)
{
        for (int i = 0; i < NXYZ; i++) {
                data[i].x = 0 ;
                data[i].y= 0;
        }
}


void printwholeres(float2 *out)
{
	printf("=====================\n");
	for(int i=0;i<10;i++)
		printf("%f %f \n",out[i].x,out[i].y);  //a
	printf("\n");
	for(int i=0+(NX*NY/2);i<10+(NX*NY/2);i++)  //b
		printf("%f %f \n",out[i].x,out[i].y);
	printf("\n");
	for(int i=NX*(NY-1);i<NX*(NY-1)+10;i++) //c
		printf("%f %f \n",out[i].x,out[i].y);
	printf("\n");
}

void printfalldata(float2 *din,int dimall)
{
	for(int i=0;i<dimall;i++)
	{
		printf("%f %f \n",din[i].x,din[i].y);
	}
}

__global__ void setdata(float2 *din,int numElements)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements)
    {
        //C[i].x = A[i] * B[i];
        din[i].x=i;din[i].y=0;
    }
}


int main()
{

	size_t size = 1024 * sizeof(float);
	int deviceCount;
	hipGetDeviceCount(&deviceCount);

	deviceCount=1;
	float2 *f , *out ;
	
    hipHostMalloc( (void **)&f, sizeof(float2) * NXYZ,hipHostMallocDefault);
	hipHostMalloc( (void **)&out, sizeof(float2) * NXYZ,hipHostMallocDefault);
	datainitzero(f);
	for(int i=0;i<deviceCount;i++)
	{
		hipSetDevice(i);

        float2 *d_in ;
        hipMalloc((void**) & (d_in), sizeof(float2) * NX*NY*NZ);
        hipMemcpy(d_in, f, NX*NY*NZ * sizeof(float2), hipMemcpyHostToDevice);
		
		int threadsPerBlock = 256;
		int blocksPerGrid =(NX*NY*NZ/2 + threadsPerBlock - 1) / threadsPerBlock;
		
		
		hipLaunchKernelGGL(setdata, 
                  blocksPerGrid,
                  threadsPerBlock,
                  0, 0,
                  d_in ,NX*NY*NZ/2);
		hipMemcpy(out, d_in, NX*NY*NZ * sizeof(float2), hipMemcpyDeviceToHost);
		hipDeviceSynchronize();
	
	}
	printfalldata(out,NX*NY*NZ);

	
	return 0;
}
