#include "stdio.h"
#include "hip/hip_runtime_api.h"
#include "hip/hip_vector_types.h"
#include "rocfft.h"


//#define PRINT

#define REP_TIMES 100

float testmoduleGPU(int dimx,int dimy) {
	int N[2];
	N[0] = dimx, N[1] = dimy;
	int LENGTH = N[0] * N[1];
	float2 *input = (float2*) malloc(LENGTH * sizeof(float2));
	float2 *output = (float2*) malloc(
			LENGTH * sizeof(float2));
	int i;
	for (i = 0; i < N[0] * N[1]; i++) {
		input[i].x = i % 1000;
		input[i].y = 0;
	}


	rocfft_setup();

	float2 *d_inputData, *d_outData;
	hipMalloc((void**) &d_inputData, N[0] * N[1] * sizeof(float2));
	hipMalloc((void**) &d_outData, N[0] * N[1] * sizeof(float2));

	hipMemcpy(d_inputData, input, N[0] * N[1] * sizeof(float2),
			hipMemcpyHostToDevice);

    // Create rocFFT plan
    rocfft_plan plan = NULL;
    size_t *lengths= (size_t *)malloc(sizeof(size_t)*2);
    lengths[0]=N[0];
    lengths[1]=N[1];

    rocfft_plan_create(&plan, rocfft_placement_notinplace,
         rocfft_transform_type_complex_forward, rocfft_precision_single,
         2, lengths, 1, NULL);

    size_t fbuffersize = 0;

    rocfft_plan_get_work_buffer_size(plan, &fbuffersize);
 	printf("worksize : %ld and complex size %ld \n",fbuffersize,N[0]*N[1]*sizeof(float2));


 	rocfft_execution_info forwardinfo = NULL;
 	rocfft_execution_info_create(&forwardinfo);


    void* fbuffer = NULL;
    hipMalloc(&fbuffer, fbuffersize);
    rocfft_execution_info_set_work_buffer(forwardinfo, fbuffer, fbuffersize);



	hipEvent_t start1;
	hipEventCreate(&start1);
	hipEvent_t stop1;
	hipEventCreate(&stop1);
	hipEventRecord(start1, NULL);
	for (int i = 0; i < 100; i++) {
    rocfft_execute(plan,(void**) &d_inputData, (void**)&d_outData, forwardinfo);

	}

	hipEventRecord(stop1, NULL);
	hipEventSynchronize(stop1);
	float msecTotal1 = 0.0f;
	hipEventElapsedTime(&msecTotal1, start1, stop1);
	hipMemcpy(output, d_outData, LENGTH * sizeof(float2), hipMemcpyDeviceToHost);


    // Destroy plan
    rocfft_plan_destroy(plan);

	rocfft_cleanup();
	free(input);
	free(output);
	hipFree(d_inputData);
	hipFree(d_outData);
	return msecTotal1;
}

int main() {
	double timeres[200];
    //128=2^7    ; 8192=2^13
	int pownum=3;
	//for(pownum=7;pownum<=13;pownum++)
	{
		double avertime = 0;
		for (int i = 0; i < 100; i++) {
			timeres[i] = testmoduleGPU(800,800);
			printf("ITER %f \n", timeres[i]);
			avertime += timeres[i];
		}
		printf("\n AVER %f \n", avertime / 100);
	}
}

