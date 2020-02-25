
//#include <iostream>
//using namespace std;
#include <stdio.h>
#include <hip/hip_runtime.h>
#include "hip/hip_runtime_api.h"
#include "hip/hip_vector_types.h"
#include "rocfft.h"


#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

#define NX 720
#define NY 720
#define NZ 720
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
	int device;
	for (device = 0; device < deviceCount; ++device) {
		hipDeviceProp_t  deviceProp;
		hipGetDeviceProperties(&deviceProp, device);
		printf("Device %d has compute capability %d.%d.\n",device, deviceProp.major, deviceProp.minor);
		printf("%d %d \n", HIP_VERSION_MAJOR,HIP_VERSION_MINOR );
	}
	if(deviceCount>=1)
	{
		int can_access_peer = -100;
		hipDeviceCanAccessPeer(&can_access_peer, 0, 1);
		printf("access 0-1: %d \n",can_access_peer);can_access_peer = -100;
		hipDeviceCanAccessPeer(&can_access_peer, 0, 2);
		printf("access 0-2: %d \n",can_access_peer);
		hipSetDevice(0);
        hipDeviceEnablePeerAccess(1, 0);
		HANDLE_ERROR(hipDeviceCanAccessPeer(&can_access_peer, 0, 1));
		printf("access 0-1: %d \n",can_access_peer);
	}

	deviceCount=2;
	float2 *f , *out ;
	//f= (float2 *)malloc(sizeof(float2)*NXYZ);
	//out= (float2 *)malloc(sizeof(float2)*NXYZ);
    hipHostMalloc( (void **)&f, sizeof(float2) * NXYZ,hipHostMallocDefault);
	hipHostMalloc( (void **)&out, sizeof(float2) * NXYZ,hipHostMallocDefault);
	      
	datainit(f);
	int offset=0;
	
	
	for(int i=0;i<deviceCount;i++)
	{
		hipSetDevice(i);

        float2 *d_in ;
        hipMalloc((void**) & (d_in), sizeof(float2) * NX*NY*NZ);
        hipMemcpy(d_in, f, NX*NY*NZ * sizeof(float2), hipMemcpyHostToDevice);
		hipDeviceSynchronize();
	
		// Create rocFFT plan

		size_t *lengths= (size_t *)malloc(sizeof(size_t)*3);
		lengths[0]=NX;
		lengths[1]=NY;
		lengths[2]=NZ/2;
		rocfft_setup();	
		rocfft_plan plan = NULL;
		rocfft_plan_create(&plan, rocfft_placement_inplace,
         rocfft_transform_type_complex_forward, rocfft_precision_single,
         3, lengths, 1, NULL);

		size_t fbuffersize = 0;
		rocfft_plan_get_work_buffer_size(plan, &fbuffersize);

		rocfft_execution_info forwardinfo = NULL;
		rocfft_execution_info_create(&forwardinfo);

		void* fbuffer = NULL;
		hipMalloc(&fbuffer, fbuffersize);
		rocfft_execution_info_set_work_buffer(forwardinfo, fbuffer, fbuffersize);
		//printf("size is : %ld \n",fbuffersize);
	
		float2 *newaddr=d_in+offset;
		rocfft_execute(plan,(void**) &(newaddr), (void**)&(newaddr), forwardinfo);
        hipMemcpy(out+offset, d_in+offset, NX*NY*NZ/2 * sizeof(float2), hipMemcpyDeviceToHost);
		offset+=NX*NY*NZ/2;
        hipFree(d_in);
        rocfft_plan_destroy(plan);
		rocfft_cleanup();	
	}
	printwholeres(out);
	printf("\n");


	
	return 0;
}
