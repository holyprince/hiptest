for 2d fft , the part is needed:


    rocfft_plan_get_work_buffer_size(plan, &fbuffersize);
 	printf("worksize : %ld and complex size %ld \n",fbuffersize,N[0]*N[1]*sizeof(float2));


 	rocfft_execution_info forwardinfo = NULL;
 	rocfft_execution_info_create(&forwardinfo);


    void* fbuffer = NULL;
    hipMalloc(&fbuffer, fbuffersize);
    rocfft_execution_info_set_work_buffer(forwardinfo, fbuffer, fbuffersize);


rocFFT link  
https://github.com/ROCmSoftwarePlatform/rocFFT  

2dstride.cpp : set stride and distance similar with cufft  
https://blog.csdn.net/wzh1026/article/details/100641403

