
#include "test_common.h"

__global__ void run_printf() { printf("Hello World\n"); }

int main() {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(run_printf), dim3(1), dim3(1), 0, 0);
    hipDeviceSynchronize();
    passed();
}
