#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
#include <string.h>
#define MAX_VALUE 10000000


void OpenMPTest()
{
    int index= 0;
    int time1 = 0;
    int time2 = 0;
    double value1 = 0.0, value2 = 0.0;
    double result[2];

    for(index = 1; index < MAX_VALUE; index ++)
        value1 += 1.0 / index;

    memset(result , 0, sizeof(double) * 2);
    omp_get_num_threads();
#pragma omp parallel for
    for(index = 0; index < 2; index++)
        result[index] = result[index];

    value2 = result[0] + result[1];

    return;
}

int main()
{
    OpenMPTest();

    return 0;
}

