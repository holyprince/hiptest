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


error: cannot take the address of an rvalue of type 'float2 *' (aka 'HIP_vector_type<float, 2> *')
[float2.cpp]

[multigpufft.cpp]
对于大尺度FFT来说 memcpy之后的同步很重要

[twogpudirectlink.cpp]
使用gpudirect需要预先设置enable，第一个版本，单节点两个GPU卡stride方法
