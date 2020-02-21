
#include <stdlib.h>
#include <stdio.h>
 
#include <string.h>
#include <math.h>
#include "timer.h"
 
#include "hip/hip_runtime_api.h"
#include "hip/hip_vector_types.h"
#include "rocfft.h"

#define Ndim 2
#define NX 10
#define NY 5
 
 
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
 
int main() {
 
	teststride1();
	teststride2();
}
/*

45.000000 0.000000
-5.000000 15.388417
-5.000000 6.881910
-5.000000 3.632713
-5.000000 1.624599
-5.000000 0.000000
-5.000000 -1.624599
-5.000000 -3.632713
-5.000000 -6.881910
-5.000000 -15.388417

145.000000 0.000000
-5.000000 15.388417
-5.000000 6.881910
-5.000000 3.632713
-5.000000 1.624599
-5.000000 0.000000
-5.000000 -1.624599
-5.000000 -3.632713
-5.000000 -6.881910
-5.000000 -15.388417

245.000000 0.000000
-5.000000 15.388417
-5.000000 6.881910
-5.000000 3.632713
-5.000000 1.624599
-5.000000 0.000000
-5.000000 -1.624599
-5.000000 -3.632713
-5.000000 -6.881910
-5.000000 -15.388417

345.000000 0.000000
-5.000000 15.388417
-5.000000 6.881910
-5.000000 3.632713
-5.000000 1.624599
-5.000000 0.000000
-5.000000 -1.624599
-5.000000 -3.632713
-5.000000 -6.881910
-5.000000 -15.388417

445.000000 0.000000
-5.000000 15.388417
-5.000000 6.881910
-5.000000 3.632713
-5.000000 1.624599
-5.000000 0.000000
-5.000000 -1.624599
-5.000000 -3.632713
-5.000000 -6.881910
-5.000000 -15.388417

100.000000 0.000000
105.000000 0.000000
110.000000 0.000000
115.000000 0.000000
120.000000 0.000000
125.000000 0.000000
130.000000 0.000000
135.000000 0.000000
140.000000 0.000000
145.000000 0.000000

-25.000000 34.409550
-25.000000 34.409550
-25.000000 34.409550
-25.000000 34.409550
-25.000000 34.409550
-25.000000 34.409550
-25.000000 34.409550
-25.000000 34.409550
-25.000000 34.409550
-25.000000 34.409550

-25.000000 8.122993
-25.000000 8.122993
-25.000000 8.122993
-25.000000 8.122993
-25.000000 8.122993
-25.000000 8.122993
-25.000000 8.122993
-25.000000 8.122993
-25.000000 8.122993
-25.000000 8.122993

-25.000000 -8.122993
-25.000000 -8.122993
-25.000000 -8.122993
-25.000000 -8.122993
-25.000000 -8.122993
-25.000000 -8.122993
-25.000000 -8.122993
-25.000000 -8.122993
-25.000000 -8.122993
-25.000000 -8.122993

-25.000000 -34.409550
-25.000000 -34.409550
-25.000000 -34.409550
-25.000000 -34.409550
-25.000000 -34.409550
-25.000000 -34.409550
-25.000000 -34.409550
-25.000000 -34.409550
-25.000000 -34.409550
-25.000000 -34.409550
*/
 
