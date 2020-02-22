
#include <stdlib.h>
#include <stdio.h>
 
#include <string.h>
#include <math.h>
#include "timer.h"
 
#include "hip/hip_runtime_api.h"
#include "hip/hip_vector_types.h"
#include "rocfft.h"


#define NX 10
#define NY 5
#define NZ 2
 
void teststride1() {
 
	int N[2];
	N[0] = NX, N[1] = NY;
	int NXY = N[0] * N[1];
	float2 *input = (float2*) malloc(NXY * sizeof(float2));
	float2 *output = (float2*) malloc(NXY * sizeof(float2));
	int i;
	for (i = 0; i < NXY; i++) {
		input[i].x = i % 1000;
		input[i].y = 0;
	}
	float2 *d_inputData, *d_outData;
	hipMalloc((void**) &d_inputData, N[0] * N[1] * sizeof(float2));
	hipMalloc((void**) &d_outData, N[0] * N[1] * sizeof(float2));
	hipMemcpy(d_inputData, input, N[0] * N[1] * sizeof(float2), hipMemcpyHostToDevice);
	size_t in_offsets[1],out_offsets[1];
	size_t in_strides_size,in_strides[1],in_distance;
	size_t out_strides_size,out_strides[1],out_distance;
	rocfft_setup();
	rocfft_plan plan;
	rocfft_plan_description description = NULL;
	rocfft_plan_description_create(&description);
	in_strides_size=1;out_strides_size=1;
	in_distance=NX;out_distance=NX;
	in_strides[0]=1;out_strides[0]=1;
	//in_strides[1]=1;out_strides[1]=1;
	rocfft_plan_description_set_data_layout(description,rocfft_array_type_complex_interleaved,rocfft_array_type_complex_interleaved,
	NULL,NULL,
	in_strides_size,in_strides,in_distance,
	out_strides_size,out_strides,out_distance);
	
	size_t *lengths= (size_t *)malloc(sizeof(size_t)*1);
	lengths[0]=NX;
	//lengths[1]=NY;
	rocfft_plan_create(&plan,rocfft_placement_inplace,rocfft_transform_type_complex_forward,rocfft_precision_single,1,lengths,NY,description);
	
	size_t fbuffersize = 0;
	rocfft_plan_get_work_buffer_size(plan, &fbuffersize);

	rocfft_execution_info forwardinfo = NULL;
	rocfft_execution_info_create(&forwardinfo);

	void* fbuffer = NULL;
	hipMalloc(&fbuffer, fbuffersize);
	rocfft_execution_info_set_work_buffer(forwardinfo, fbuffer, fbuffersize);



	rocfft_execute(plan,(void**) &d_inputData, (void**)&d_inputData, forwardinfo);
	hipMemcpy(output, d_inputData, NXY * sizeof(float2), hipMemcpyDeviceToHost);

 
	for (i = 0; i < NXY; i++) {
		if(i%NX==0)
			printf("\n");
		printf("%f %f \n", output[i].x, output[i].y);
	}
 
	rocfft_plan_destroy(plan);
	rocfft_cleanup();
	free(input);
	free(output);
	hipFree(d_inputData);
	hipFree(d_outData);
}

void teststride2() {
 
	int N[2];
	N[0] = NX, N[1] = NY;
	int NXY = N[0] * N[1];
	float2 *input = (float2*) malloc(NXY * sizeof(float2));
	float2 *output = (float2*) malloc(NXY * sizeof(float2));
	int i;
	for (i = 0; i < NXY; i++) {
		input[i].x = i % 1000;
		input[i].y = 0;
	}
	float2 *d_inputData, *d_outData;
	hipMalloc((void**) &d_inputData, N[0] * N[1] * sizeof(float2));
	hipMalloc((void**) &d_outData, N[0] * N[1] * sizeof(float2));
	hipMemcpy(d_inputData, input, N[0] * N[1] * sizeof(float2), hipMemcpyHostToDevice);
	size_t in_offsets[1],out_offsets[1];
	size_t in_strides_size,in_strides[1],in_distance;
	size_t out_strides_size,out_strides[1],out_distance;
	rocfft_setup();
	rocfft_plan plan;
	rocfft_plan_description description = NULL;
	rocfft_plan_description_create(&description);
	in_strides_size=1;out_strides_size=1;
	in_distance=1;out_distance=1;
	in_strides[0]=NX;out_strides[0]=NX;
	rocfft_plan_description_set_data_layout(description,rocfft_array_type_complex_interleaved,rocfft_array_type_complex_interleaved,
	NULL,NULL,
	in_strides_size,in_strides,in_distance,
	out_strides_size,out_strides,out_distance);
	
	size_t *lengths= (size_t *)malloc(sizeof(size_t)*1);
	lengths[0]=NY;
	rocfft_plan_create(&plan,rocfft_placement_inplace,rocfft_transform_type_complex_forward,rocfft_precision_single,1,lengths,NX,description);
	
	size_t fbuffersize = 0;
	rocfft_plan_get_work_buffer_size(plan, &fbuffersize);

	rocfft_execution_info forwardinfo = NULL;
	rocfft_execution_info_create(&forwardinfo);

	void* fbuffer = NULL;
	hipMalloc(&fbuffer, fbuffersize);
	rocfft_execution_info_set_work_buffer(forwardinfo, fbuffer, fbuffersize);



	rocfft_execute(plan,(void**) &d_inputData, (void**)&d_inputData, forwardinfo);
	hipMemcpy(output, d_inputData, NXY * sizeof(float2), hipMemcpyDeviceToHost);

 
	for (i = 0; i < NXY; i++) {
		if(i%NX==0)
			printf("\n");
		printf("%f %f \n", output[i].x, output[i].y);
	}
 
	rocfft_plan_destroy(plan);
	rocfft_cleanup();
	free(input);
	free(output);
	hipFree(d_inputData);
	hipFree(d_outData);
}

void teststride3() {
 
	int N[3];
	N[0] = NX, N[1] = NY; N[2] = NZ;
	int NXYZ = N[0] * N[1]* N[2];
	float2 *input = (float2*) malloc(NXYZ * sizeof(float2));
	float2 *output = (float2*) malloc(NXYZ * sizeof(float2));
	int i;
	for (i = 0; i < NXYZ; i++) {
		input[i].x = i % 1000;
		input[i].y = 0;
	}
	float2 *d_inputData, *d_outData;
	hipMalloc((void**) &d_inputData, NXYZ * sizeof(float2));
	hipMalloc((void**) &d_outData, NXYZ * sizeof(float2));
	hipMemcpy(d_inputData, input, NXYZ * sizeof(float2), hipMemcpyHostToDevice);

	size_t in_strides_size,in_strides[1],in_distance;
	size_t out_strides_size,out_strides[1],out_distance;
	rocfft_setup();
	rocfft_plan plan;
	rocfft_plan_description description = NULL;
	rocfft_plan_description_create(&description);
	in_strides_size=1;out_strides_size=1;
	in_distance=NX*NY;out_distance=NX*NY;
	in_strides[0]=1;out_strides[0]=1;
	//in_strides[1]=NX;out_strides[1]=NX;
	rocfft_plan_description_set_data_layout(description,rocfft_array_type_complex_interleaved,rocfft_array_type_complex_interleaved,
	NULL,NULL,
	in_strides_size,in_strides,in_distance,
	out_strides_size,out_strides,out_distance);
	
	size_t *lengths= (size_t *)malloc(sizeof(size_t)*2);
	lengths[0]=NX;
	lengths[1]=NY;
	rocfft_plan_create(&plan,rocfft_placement_inplace,rocfft_transform_type_complex_forward,rocfft_precision_single,2,lengths,NZ,description);
	
	size_t fbuffersize = 0;
	rocfft_plan_get_work_buffer_size(plan, &fbuffersize);

	rocfft_execution_info forwardinfo = NULL;
	rocfft_execution_info_create(&forwardinfo);

	void* fbuffer = NULL;
	hipMalloc(&fbuffer, fbuffersize);
	rocfft_execution_info_set_work_buffer(forwardinfo, fbuffer, fbuffersize);



	rocfft_execute(plan,(void**) &d_inputData, (void**)&d_inputData, forwardinfo);
	hipMemcpy(output, d_inputData, NXYZ * sizeof(float2), hipMemcpyDeviceToHost);

 
	for (i = 0; i < NXYZ; i++) {
		if(i%NX==0)
			printf("\n");
		if(i%NX*NY==0)
			printf("\n");
		printf("%f %f \n", output[i].x, output[i].y);
	}
 
	rocfft_plan_destroy(plan);
	rocfft_cleanup();
	free(input);
	free(output);
	hipFree(d_inputData);
	hipFree(d_outData);
}
int main() {
 
	//teststride1();
	//teststride2();
	teststride3();
}

 
