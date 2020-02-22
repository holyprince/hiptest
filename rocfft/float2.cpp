
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





int main()
{
	float2 *inputdata;
	inputdata=(float2 *)malloc(sizeof(float2)*100);
	for(int i=0;i<100;i++)
	{
		inputdata[i].x=i;
		inputdata[i].y=0;
	}
	
	printf("%f %f \n",inputdata[10].x,inputdata[11].x);
	
	printf("%f %f \n",(inputdata[10]+10).x,(inputdata[11]+20).x);

	return 0;
}
