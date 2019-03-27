#include <stdio.h>
#include <hip/hip_runtime.h>
#include "acc_projectorkernel_impl.h"







//template<bool REF3D, bool DATA3D, int block_sz, int eulers_per_block, int prefetch_fraction>
template<bool REF3D>
__global__ void hip_kernel_diff2_coarse(
		float *g_eulers,
		float *trans_x,
		float *trans_y,
		float *trans_z,
		float *g_real,
		float *g_imag, 
		AccProjectorKernel projector,
		float *g_corr,
		float *g_diff2s,
		int translation_num,
		int image_size
		)
{}

int main()
{

	bool REF3D2=true;
	//	,DATA3D;
	int block_sz, eulers_per_block, prefetch_fr;
	XFLOAT* init;
	AccProjectorKernel projector(1,2,3,4,5,6,7,8,9,10,init);
	float *g_eulers,*trans_x,*trans_y,*trans_z,*g_real,*g_imag,*g_corr,*g_diff2s;
	int translation_num,image_size ;

	//hipLaunchKernelGGL(hip_kernel_diff2_coarse<REF3D, DATA3D, block_sz, eulers_per_block, prefetch_fr>,128,128,0,0,image_size
	hipLaunchKernelGGL(hip_kernel_diff2_coarse <true> ,128,128,0,0,
			g_eulers,
			trans_x,
			trans_y,
			trans_z,
			g_real,
			g_imag,
			projector,
			g_corr,
			g_diff2s,
			translation_num,
			image_size);
	return 0;
}

