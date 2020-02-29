#include <stdlib.h>
#include <stdio.h>

#include <string.h>
#include <math.h>
#include "timer.h"

#include "hip/hip_runtime_api.h"
#include "hip/hip_vector_types.h"
#include "rocfft.h"

#define Ndim 3
#define NX 720
#define NY 720
#define NZ 720
#define NXYZ NX*NY*NZ
#define NXYhalfZ NX*NY*(NZ/2)
#define MAXGPU 100

typedef struct
{
    //Host-side input data
    int datasize; // ful size and the real size is half

    int devicenum;

    int selfoffset;

    float2 *h_Data;

    //Device buffers
    float2 *d_Data;

} MultiGPUplan;

void datainit(float2 *data)
{
        for (int i = 0; i < NXYZ; i++) {
                data[i].x = i % 5000 ;
                data[i].y= 0;
        }
}



void printressingle(float2 *data)
{
        for(int i=0;i<10;i++)
        {
                printf("\n");
                for(int j=0;j<NZ;j++)
                        printf("%f %f \n",data[i+j*NX*NY].x,data[i+j*NX*NY].y);
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



void whole3dfft(float2 *f,float2 *out)
{
        float2 *d_in ;
        hipMalloc((void**) & (d_in), sizeof(float2) * NX*NY*NZ);
        hipMemcpy(d_in, f, NX*NY*NZ * sizeof(float2), hipMemcpyHostToDevice);
		
		// Create rocFFT plan
		rocfft_setup();
		rocfft_plan plan = NULL;
		size_t *lengths= (size_t *)malloc(sizeof(size_t)*3);
		lengths[0]=NX;
		lengths[1]=NY;
		lengths[1]=NZ;

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



		rocfft_execute(plan,(void**) &d_in, (void**)&d_in, forwardinfo);

        hipMemcpy(out, d_in, NX*NY*NZ * sizeof(float2), hipMemcpyDeviceToHost);

        printwholeres(out);

        hipFree(d_in);
        rocfft_plan_destroy(plan);
		rocfft_cleanup();

}

void dividefft(float2 *f,float2 *out) // test first do 2D fft and do 1 dim fft
{

    int res;
    float2 *d_in ;
    hipMalloc((void**) & (d_in), sizeof(float2) * NX*NY*NZ);
    hipMemcpy(d_in, f, NX*NY*NZ * sizeof(float2), hipMemcpyHostToDevice);

/////////////////////////////////////////////////////

	size_t in_strides_size,in_strides[2],in_distance;
	size_t out_strides_size,out_strides[2],out_distance;
	rocfft_setup();
	rocfft_plan plan;
	rocfft_plan_description description = NULL;
	rocfft_plan_description_create(&description);
	in_strides_size=2;out_strides_size=2;
	in_distance=NX*NY;out_distance=NX*NY;
	in_strides[0]=1;out_strides[0]=1;
	in_strides[1]=NX;out_strides[1]=NX;
	rocfft_plan_description_set_data_layout(description,
	rocfft_array_type_complex_interleaved,rocfft_array_type_complex_interleaved,
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



	rocfft_execute(plan,(void**) &d_in, (void**)&d_in, forwardinfo);
	printf("2D res :\n");
    hipMemcpy(out, d_in, NX*NY*NZ * sizeof(float2), hipMemcpyDeviceToHost);
    printwholeres(out);
	
	rocfft_plan_destroy(plan);
	rocfft_cleanup();
/////////////////////
   
	rocfft_setup();
	size_t in_zstrides_size,in_zstrides[1],in_zdistance;
	size_t out_zstrides_size,out_zstrides[1],out_zdistance;
	rocfft_plan zplan;
	rocfft_plan_description zdescription = NULL;
	rocfft_plan_description_create(&zdescription);
	in_zstrides_size=1;out_zstrides_size=1;
	in_zstrides[0]=NX*NY;out_zstrides[0]=NX*NY;
	in_zdistance=1;out_zdistance=1;
	rocfft_plan_description_set_data_layout(zdescription,
	rocfft_array_type_complex_interleaved,rocfft_array_type_complex_interleaved,
	NULL,NULL,
	in_zstrides_size,in_zstrides,in_zdistance,
	out_zstrides_size,out_zstrides,out_zdistance);
	
	size_t *zlengths= (size_t *)malloc(sizeof(size_t)*1);
	zlengths[0]=NZ;
	rocfft_plan_create(&zplan,rocfft_placement_inplace,rocfft_transform_type_complex_forward,rocfft_precision_single,1,zlengths,NX*NY,zdescription);
	
	size_t zfbuffersize = 0;
	rocfft_plan_get_work_buffer_size(zplan, &zfbuffersize);

	rocfft_execution_info zforwardinfo = NULL;
	rocfft_execution_info_create(&zforwardinfo);

	void* zfbuffer = NULL;
	hipMalloc(&zfbuffer, zfbuffersize);
	rocfft_execution_info_set_work_buffer(zforwardinfo, zfbuffer, zfbuffersize);


	rocfft_execute(zplan,(void**) &d_in, (void**)&d_in, zforwardinfo);	
        

        //printf("check4: %d\n",res);
	printf("3D res :\n");
    hipMemcpy(out, d_in, NX*NY*NZ * sizeof(float2), hipMemcpyDeviceToHost);
    printwholeres(out);

    hipFree(d_in);
	//rocfft_plan_destroy(plan);
	//rocfft_plan_destroy(zplan);
	rocfft_cleanup();

}

void printtransres(float2 *out)
{
        // int blocknum=NX*NY*NZ/2; next block
        int blocknum =0 ;
        int offset;
        printf("=====================\n");
        for(int i=0+blocknum;i<10+blocknum;i++)
                printf("%.2f ",out[i].x);
        printf("\n");
        for(int i=NX+blocknum;i<NX+10+blocknum;i++)
                printf("%.2f ",out[i].x);
        printf("\n");
        for(int i=NX*NY+blocknum;i<NX*NY+10+blocknum;i++)
                printf("%.2f ",out[i].x);
        printf("\n");

        offset = NX*NY/2;
        for(int i=0+offset+blocknum;i<10+offset+blocknum;i++)
                printf("%.2f ",out[i].x);
        printf("\n");
        for(int i=NX+offset+blocknum;i<NX+10+offset+blocknum;i++)
                printf("%.2f ",out[i].x);
        printf("\n");
        for(int i=NX*NY+offset+blocknum;i<NX*NY+10+offset+blocknum;i++)
                printf("%.2f ",out[i].x);
        printf("\n");
}
int main()
{
        float2 *f , *out ;
        hipHostMalloc( (void **)&f, sizeof(float2) * NXYZ,hipHostMallocDefault);
        hipHostMalloc( (void **)&out, sizeof(float2) * NXYZ,hipHostMallocDefault);

        datainit(f);

		
		
		int GPU_N;
        int deviceNum[MAXGPU];
        int err_info;
        GPU_N = 2;
        MultiGPUplan plan[MAXGPU];
        for (int i = 0; i < GPU_N; i++) {
                deviceNum[i] = i;
                plan[i].devicenum = i;
                plan[i].datasize = NXYZ;
        }
        plan[0].selfoffset = 0;
        plan[1].selfoffset = NX * NY * NZ / 2;		
		
        int can_access_peer = -100;
		hipDeviceCanAccessPeer(&can_access_peer, plan[0].devicenum,plan[1].devicenum);
			for (int i = 0; i < GPU_N; i++) {
					hipSetDevice(plan[i].devicenum);
					hipDeviceEnablePeerAccess((GPU_N - 1) - plan[i].devicenum, 0);
			}
			for (int i = 0; i < GPU_N; i++) {
					hipSetDevice(plan[i].devicenum);
					hipDeviceSynchronize();
			}






        for (int temp = 0; temp < 1; temp++) {
                timeval timertemp1;
                gettimeofday(&timertemp1, NULL);
                StartTimer();
// setp 1 : malloc
               int offset = 0;
                for (int i = 0; i < GPU_N; ++i) {
                        err_info = hipSetDevice(plan[i].devicenum);
                        err_info = hipMalloc((void**) &(plan[i].d_Data), sizeof(float2) * plan[i].datasize);
                        err_info = hipMemcpyAsync(plan[i].d_Data + plan[i].selfoffset, f + offset, (plan[i].datasize / 2) * sizeof(float2),hipMemcpyHostToDevice);
						
                        offset += NXYhalfZ;
						hipDeviceSynchronize();
                }
        
             	printf("step1:cpy  : %.3f\n", GetTimer());
                StartTimer();

// step 2 : do 2d C2C inplace  fft

				offset=0;
		
                for (int i = 0; i < GPU_N; i++) {
                        hipSetDevice(plan[i].devicenum);

							size_t in_strides_size,in_strides[2],in_distance;
							size_t out_strides_size,out_strides[2],out_distance;
							size_t *lengths= (size_t *)malloc(sizeof(size_t)*2);
							lengths[0]=NX;
							lengths[1]=NY;
							in_strides_size=2;out_strides_size=2;
							in_distance=NX*NY;out_distance=NX*NY;
							in_strides[0]=1;out_strides[0]=1;
							in_strides[1]=NX;out_strides[1]=NX;
							rocfft_setup();
							rocfft_plan xyplan;
							rocfft_plan_description description = NULL;
							rocfft_plan_description_create(&description);
							rocfft_plan_description_set_data_layout(description,
							rocfft_array_type_complex_interleaved,rocfft_array_type_complex_interleaved,
							NULL,NULL,
							in_strides_size,in_strides,in_distance,
							out_strides_size,out_strides,out_distance);
							
							rocfft_plan_create(&xyplan,rocfft_placement_inplace,rocfft_transform_type_complex_forward,rocfft_precision_single,2,lengths,NZ/2,description);
							size_t fbuffersize = 0;
							rocfft_plan_get_work_buffer_size(xyplan, &fbuffersize);
							rocfft_execution_info forwardinfo = NULL;
							rocfft_execution_info_create(&forwardinfo);
							void* fbuffer = NULL;
							hipMalloc(&fbuffer, fbuffersize);
							rocfft_execution_info_set_work_buffer(forwardinfo, fbuffer, fbuffersize);
							printf("fbuffersize : %ld\n",fbuffersize);
							hipDeviceSynchronize();
							//rocfft_execute(xyplan,(void**)&(plan[i].d_Data + plan[i].selfoffset),(void**)&(plan[i].d_Data + plan[i].selfoffset),forwardinfo);
							float2 *newaddr1=plan[i].d_Data;
							newaddr1=plan[i].d_Data + plan[i].selfoffset;
							rocfft_execute(xyplan,(void**)&(newaddr1),(void**)&(newaddr1),forwardinfo);
							hipMemcpy(out + offset,newaddr1,(plan[i].datasize / 2) * sizeof(float2),hipMemcpyDeviceToHost);							
							rocfft_plan_destroy(xyplan);
							hipFree(fbuffer);
							rocfft_cleanup();
							offset+=NXYhalfZ;
		
                }
                for (int i = 0; i < GPU_N; i++) {
                        hipSetDevice(plan[i].devicenum);
                        hipDeviceSynchronize();
                }
                printf("2dfft: %.3f\n", GetTimer());
                StartTimer();



		


// step 3 : do data change on two gpu
 
                StartTimer();
                // step 4 : move data to GPU2  and gpu1 to gpu 2
                //hipSetDevice(0);
                int halfslice = (NY / 2) * NX;
                int sliceoffset;
                for (int j = 0; j < NZ / 2; j++) {
                        sliceoffset = halfslice + j * NX * NY;
                        hipMemcpy(plan[1].d_Data + sliceoffset,
                                        plan[0].d_Data + sliceoffset,
                                        halfslice * sizeof(float2), hipMemcpyDefault);
                }
                for (int j = 0; j < NZ / 2; j++) {
                        sliceoffset = NX * NY * NZ / 2 + j * NX * NY;
                        hipMemcpy(plan[0].d_Data + sliceoffset,
                                        plan[1].d_Data + sliceoffset,
                                        halfslice * sizeof(float2), hipMemcpyDefault);
                }
                for (int i = 0; i < GPU_N; i++) {
                        hipSetDevice(plan[i].devicenum);
                        hipDeviceSynchronize();
                }
                printf("allto: %.3f\n", GetTimer());
                StartTimer();


 


                //step 5:  do the rest fft



                int fftoffset[2];
                fftoffset[0] = 0;
                fftoffset[1] = NX * NY / 2;
				
				
				size_t in_zstrides_size,in_zstrides[1],in_zdistance;
				size_t out_zstrides_size,out_zstrides[1],out_zdistance;
				in_zstrides_size=1;out_zstrides_size=1;
				in_zstrides[0]=NX*NY;out_zstrides[0]=NX*NY;
				in_zdistance=1;out_zdistance=1;
				size_t *zlengths= (size_t *)malloc(sizeof(size_t)*1);
				zlengths[0]=NZ;
                for (int i = 0; i < GPU_N; i++) {
                        hipSetDevice(plan[i].devicenum);
						rocfft_plan zplan;
						rocfft_plan_description zdescription = NULL;
						rocfft_plan_description_create(&zdescription);
						rocfft_plan_description_set_data_layout(zdescription,
							rocfft_array_type_complex_interleaved,rocfft_array_type_complex_interleaved,
							NULL,NULL,
							in_zstrides_size,in_zstrides,in_zdistance,
							out_zstrides_size,out_zstrides,out_zdistance);
						
						rocfft_plan_create(&zplan,rocfft_placement_inplace,rocfft_transform_type_complex_forward,rocfft_precision_single,1,zlengths,(NX * NY) / 2,zdescription);
							
						size_t zfbuffersize = 0;
						rocfft_plan_get_work_buffer_size(zplan, &zfbuffersize);
						rocfft_execution_info zforwardinfo = NULL;
						rocfft_execution_info_create(&zforwardinfo);
						void* zfbuffer = NULL;
						hipMalloc(&zfbuffer, zfbuffersize);
						rocfft_execution_info_set_work_buffer(zforwardinfo, zfbuffer, zfbuffersize);
						float2 *newaddr2=plan[i].d_Data + fftoffset[i];
						rocfft_execute(zplan,(void**) &(newaddr2), (void**)&(newaddr2), zforwardinfo);
						rocfft_plan_destroy(zplan);	
						hipFree(zfbuffer);	
						rocfft_cleanup();							
                }
                for (int i = 0; i < GPU_N; i++) {
                        hipSetDevice(plan[i].devicenum);
                        hipDeviceSynchronize();
                }
                printf("1dFFT: %.3f\n", GetTimer());
				StartTimer();
                //ALL To ALL

                for (int i = 0; i < GPU_N; ++i) {
                        err_info = hipSetDevice(plan[i].devicenum);
						int slicesize=NX*NY/2;
						int sliceoffset;
						if(i==0)
							sliceoffset=0;
						if(i==1)
							sliceoffset=NX*NY/2;
						for(int j=0;j<NZ;j++)
                        {
							hipMemcpyAsync(out + sliceoffset,plan[i].d_Data + sliceoffset,slicesize * sizeof(float2),hipMemcpyDeviceToHost);
							sliceoffset += NX*NY;
						}
                }
                for (int i = 0; i < GPU_N; i++) {
                        hipSetDevice(plan[i].devicenum);
                        hipDeviceSynchronize();
                }

				printf("final reduce data : %.3f\n", GetTimer());
                StartTimer();
                for (int i = 0; i < GPU_N; i++) {
                        hipSetDevice(plan[i].devicenum);
                        hipFree(plan[i].d_Data);
                        hipDeviceSynchronize();
                }
                printf("Free : %.3f\n", GetTimer());
                struct timeval timerStop2, timerElapsed2;
                gettimeofday(&timerStop2, NULL);
                timersub(&timerStop2, &timertemp1, &timerElapsed2);
                printf("sum t: %.3f\n", timerElapsed2.tv_sec * 1000.0 + timerElapsed2.tv_usec / 1000.0);
                printf("==============\n");
				printwholeres(out);
			
        }
	
        hipHostFree(f);
        hipHostFree(out);

        return 0;

}
