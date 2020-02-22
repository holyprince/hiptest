
//#include <iostream>
//using namespace std;
#include <stdio.h>
#include <hip/hip_runtime.h>

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

static void HandleError( hipError_t err, const char *file, int line )
{

    if (err != hipSuccess)
    {
    	fprintf(stderr, "ERROR: %s in %s at line %d (error-code %d)\n",
						hipGetErrorString( err ), file, line, err );
		fflush(stdout);
    }
}


void testdevice()
{

	int *p0;
	int size=1024*sizeof(int);
    HANDLE_ERROR(hipSetDevice(0));
	hipMalloc(&p0, size);
    HANDLE_ERROR(hipSetDevice(2));
	hipMalloc(&p0, size);
	HANDLE_ERROR(hipSetDevice(-100));


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

	
	return 0;
}
