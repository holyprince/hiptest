#include <stdio.h>
#include <hip/hip_runtime.h>
#include "acc_projectorkernel_impl.h"







//template<bool REF3D, bool DATA3D, int block_sz, int eulers_per_block, int prefetch_fraction
//namespace AccUtilities




template< typename T1, typename T2 >
static inline
__device__ int ceilfracf(T1 a, T2 b)
{
//	return __float2int_ru(__fdividef( (float)a, (float)b ) );
	return (int)(a/b + 1);
}





template<bool REF3D, bool DATA3D, int block_sz, int eulers_per_block, int prefetch_fraction>
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

{
	int tid = threadIdx.x;

	//Prefetch euler matrices
	__shared__ XFLOAT s_eulers[eulers_per_block * 9];

	int max_block_pass_euler( ceilfracf(eulers_per_block*9, block_sz) * block_sz);

	for (int i = tid; i < max_block_pass_euler; i += block_sz)
		if (i < eulers_per_block * 9)
			s_eulers[i] = g_eulers[blockIdx.x * eulers_per_block * 9 + i];


	//Setup variables
	__shared__ XFLOAT s_ref_real[block_sz/prefetch_fraction * eulers_per_block];
	__shared__ XFLOAT s_ref_imag[block_sz/prefetch_fraction * eulers_per_block];

	__shared__ XFLOAT s_real[block_sz];
	__shared__ XFLOAT s_imag[block_sz];
	__shared__ XFLOAT s_corr[block_sz];

	XFLOAT diff2s[eulers_per_block] = {0.f};

	XFLOAT tx = trans_x[tid%translation_num];
	XFLOAT ty = trans_y[tid%translation_num];
	XFLOAT tz = trans_z[tid%translation_num];

	//Step through data
	int max_block_pass_pixel( ceilfracf(image_size,block_sz) * block_sz );

	for (int init_pixel = 0; init_pixel < max_block_pass_pixel; init_pixel += block_sz/prefetch_fraction)
	{
		/*__syncthreads();

		//Prefetch block-fraction-wise
		if(init_pixel + tid/prefetch_fraction < image_size)
		{
			int x,y,z,xy;
			if(DATA3D)
			{
				z =  floorfracf(init_pixel + tid/prefetch_fraction, projector.imgX*projector.imgY);
				xy = (init_pixel + tid/prefetch_fraction) % (projector.imgX*projector.imgY);
				x =             xy  % projector.imgX;
				y = floorfracf( xy,   projector.imgX);
				if (z > projector.maxR)
					z -= projector.imgZ;
			}
			else
			{
				x =           ( init_pixel + tid/prefetch_fraction) % projector.imgX;
				y = floorfracf( init_pixel + tid/prefetch_fraction  , projector.imgX);
			}
			if (y > projector.maxR)
				y -= projector.imgY;

//			#pragma unroll
			for (int i = tid%prefetch_fraction; i < eulers_per_block; i += prefetch_fraction)
			{
				if(DATA3D) // if DATA3D, then REF3D as well.
					projector.project3Dmodel(
						x,y,z,
						s_eulers[i*9  ],
						s_eulers[i*9+1],
						s_eulers[i*9+2],
						s_eulers[i*9+3],
						s_eulers[i*9+4],
						s_eulers[i*9+5],
						s_eulers[i*9+6],
						s_eulers[i*9+7],
						s_eulers[i*9+8],
						s_ref_real[eulers_per_block * (tid/prefetch_fraction) + i],
						s_ref_imag[eulers_per_block * (tid/prefetch_fraction) + i]);
				else if(REF3D)
					projector.project3Dmodel(
						x,y,
						s_eulers[i*9  ],
						s_eulers[i*9+1],
						s_eulers[i*9+3],
						s_eulers[i*9+4],
						s_eulers[i*9+6],
						s_eulers[i*9+7],
						s_ref_real[eulers_per_block * (tid/prefetch_fraction) + i],
						s_ref_imag[eulers_per_block * (tid/prefetch_fraction) + i]);
				else
					projector.project2Dmodel(
						x,y,
						s_eulers[i*9  ],
						s_eulers[i*9+1],
						s_eulers[i*9+3],
						s_eulers[i*9+4],
						s_ref_real[eulers_per_block * (tid/prefetch_fraction) + i],
						s_ref_imag[eulers_per_block * (tid/prefetch_fraction) + i]);
			}
		}

		//Prefetch block-wise
		if (init_pixel % block_sz == 0 && init_pixel + tid < image_size)
		{
			s_real[tid] = g_real[init_pixel + tid];
			s_imag[tid] = g_imag[init_pixel + tid];
			s_corr[tid] = g_corr[init_pixel + tid] / 2;
		}

		__syncthreads();
*/
		if (tid/translation_num < block_sz/translation_num) // NOTE int division A/B==C/B !=> A==C
		for (int i = tid / translation_num;
				i < block_sz/prefetch_fraction;
				i += block_sz/translation_num)
		{
/*			if((init_pixel + i) >= image_size) break;

			int x,y,z,xy;
			if(DATA3D)
			{
				z =  floorfracf( init_pixel + i   ,  projector.imgX*projector.imgY); //TODO optimize index extraction.
				xy =           ( init_pixel + i ) % (projector.imgX*projector.imgY);
				x =             xy  % projector.imgX;
				y = floorfracf( xy,   projector.imgX);
				if (z > projector.maxR)
					z -= projector.imgZ;
			}
			else
			{
				x =           ( init_pixel + i ) % projector.imgX;
				y = floorfracf( init_pixel + i   , projector.imgX);
			}
			if (y > projector.maxR)
				y -= projector.imgY;

			XFLOAT real, imag;

			if(DATA3D)
				translatePixel(x, y, z, tx, ty, tz, s_real[i + init_pixel % block_sz], s_imag[i + init_pixel % block_sz], real, imag);
			else
				translatePixel(x, y,    tx, ty,     s_real[i + init_pixel % block_sz], s_imag[i + init_pixel % block_sz], real, imag);

*/
			#pragma unroll
			for (int j = 0; j < eulers_per_block; j ++)
			{
				XFLOAT diff_real =  s_ref_real[eulers_per_block * i + j]; 
				diff2s[j] += diff_real ;
			}
		}
	}

	//Set global
	#pragma unroll
	for (int i = 0; i < eulers_per_block; i ++)
		atomicAdd(&g_diff2s[(blockIdx.x * eulers_per_block + i) * translation_num + tid % translation_num], diff2s[i]);



}



template<bool REF3D, bool DATA3D, int block_sz, int eulers_per_block, int prefetch_fraction>
void hip_kernel_diff2_coarsecpu(
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
{
	 hipLaunchKernelGGL(hip_kernel_diff2_coarse<REF3D,DATA3D,block_sz,eulers_per_block,prefetch_fraction>,128,128,0,0,g_eulers,trans_x,trans_y,*trans_z,g_real,g_imag,projector,g_corr,g_diff2s,translation_num,image_size);
}



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
    hip_kernel_diff2_coarsecpu<true,true,36,64,128>(g_eulers,trans_x,trans_y,trans_z,g_real,g_imag,projector,g_corr,g_diff2s,translation_num,image_size);
   //	hipLaunchKernelGGL(hip_kernel_diff2_coarse <true> ,128,128,0,0,
/*(	hipLaunchKernelGGL(hip_kernel_diff2_coarse <true> ,128,128,0,0,		g_eulers,
			trans_x,
			trans_y,
			trans_z,
			g_real,
			g_imag,
			projector,
			g_corr,
			g_diff2s,
			translation_num,
			image_size);*/
	return 0;
}

