#include "hip/hip_runtime.h"

template <bool flagt>
__global__ void constKernel(int* i) {
  // Nothing
  if(flagt)
  {
  }
  else
  {
  }
}

template<bool check_max_r2>
__global__ void hip_kernel_window_fourier_transform(
		float2 *g_in,
		float2 *g_out,
		size_t iX, size_t iY, size_t iZ, size_t iYX, //Input dimensions
		size_t oX, size_t oY, size_t oZ, size_t oYX, //Output dimensions
		size_t max_idx,
        int block_size,
        size_t max_r2 = 0
		)
{
	size_t n = hipThreadIdx_x + block_size * hipBlockIdx_x;
	size_t oOFF = oX*oY*oZ*hipBlockIdx_y;
	size_t iOFF = iX*iY*iZ*hipBlockIdx_y;
	if (n >= max_idx) return;

	long int k, i, kp, ip, jp;

	if (check_max_r2)
	{
		k = n / (iX * iY);
		i = (n % (iX * iY)) / iX;

		kp = k < iX ? k : k - iZ;
		ip = i < iX ? i : i - iY;
		jp = n % iX;

		if (kp*kp + ip*ip + jp*jp > max_r2)
			return;
	}
	else
	{
		k = n / (oX * oY);
		i = (n % (oX * oY)) / oX;

		kp = k < oX ? k : k - oZ;
		ip = i < oX ? i : i - oY;
		jp = n % oX;
	}

	long int  in_idx = (kp < 0 ? kp + iZ : kp) * iYX + (ip < 0 ? ip + iY : ip)*iX + jp;
	long int out_idx = (kp < 0 ? kp + oZ : kp) * oYX + (ip < 0 ? ip + oY : ip)*oX + jp;
	g_out[out_idx + oOFF] =  g_in[in_idx + iOFF];
}

#define WINDOW_FT_BLOCK_SIZE 128

int main() {
  int* i;
  hipMalloc(&i, sizeof(int));
  hipLaunchKernelGGL(constKernel<true>, 1, 1, 0, 0, i);
  float2*  a,b;
  size_t iX, iY, iZ,oX, oY, oZ;
  size_t max_r2=0;
  //hipLaunchKernelGGL(hip_kernel_window_fourier_transform<true>, grid_dim, WINDOW_FT_BLOCK_SIZE, 0, d_out.getStream(),
  hipLaunchKernelGGL(hip_kernel_window_fourier_transform<true>, 1, WINDOW_FT_BLOCK_SIZE, 0, 0,
                                a,
                                b,
                                iX, iY, iZ, iX * iY, //Input dimensions
                                oX, oY, oZ, oX * oY, //Output dimensions
                                iX*iY*iZ,
                                WINDOW_FT_BLOCK_SIZE,
                                max_r2);

  return 0;
}
